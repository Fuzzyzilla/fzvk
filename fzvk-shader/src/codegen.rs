//! # Codegen.
//! Parses the module and spits out the resulting tokentree
use crate::module;
use proc_macro::TokenStream;
pub fn from_spirv(spirv: &[u32]) -> TokenStream {
    let mut module = module::Module::default();
    if let Err(e) = rspirv::binary::parse_words(spirv, &mut module) {
        panic!("{e}")
    };
    // Ensure we have at least one entry point, modules are invalid without one.
    assert!(!module.entry_points.is_empty());

    // Create a struct definition for all of the specialization constants.
    let (specialization_struct_def, specialization_struct_name) = if module
        .spec_constants
        .is_empty()
    {
        // If none, definition is empty and the unit type is the constant type.
        (quote::quote! {}, quote::quote! {()})
    } else {
        // Name of our struct.
        let name = quote::format_ident!("Specialization");
        // Name -> (ty, default val) of our constants. TokenTrees/idents don't
        // impl Hash or Eq. Everything feels wrong!
        struct Field {
            constant_id: u32,
            name: proc_macro2::Ident,
            ty: proc_macro2::TokenStream,
            default_value: proc_macro2::TokenStream,
        }
        let mut fields = Vec::<Field>::default();

        for (binding_id, default_value) in &module.spec_constants {
            use module::SpecDefaultValue;
            let name = module
                .names
                .get(binding_id)
                .map(ToOwned::to_owned)
                .unwrap_or_else(|| format!("constant_{binding_id}"));
            // Convert into a rust-type, value pair.
            let (ty, default_value) = match default_value {
                SpecDefaultValue::Bool(v) => (
                    quote::quote! {::fzvk::Bool32},
                    if *v {
                        quote::quote! {::fzvk::Bool32::True}
                    } else {
                        quote::quote! {::fzvk::Bool32::False}
                    },
                ),
                SpecDefaultValue::U32(v) => (quote::quote! {u32}, quote::quote! {#v}),
                SpecDefaultValue::I32(v) => {
                    let value = i32::from_ne_bytes(v.to_ne_bytes());
                    (quote::quote! {i32}, quote::quote! {#value})
                }
                SpecDefaultValue::F32(v) => {
                    let le_bytes = v.to_le_bytes();
                    (
                        quote::quote! {f32},
                        quote::quote! {f32::from_le_bytes([#(#le_bytes),*])},
                    )
                }
            };
            let constant_id = module.decorations.get(binding_id).unwrap().spec_id.unwrap();
            fields.push(Field {
                constant_id,
                name: quote::format_ident!("{name}"),
                ty,
                default_value,
            })
        }
        // Generate the specialization map for the struct we will generate.
        // Since it is packed, the relative offset from the previous field and
        // size of fields is always four bytes.
        let mut specialization_entries = Vec::<proc_macro2::TokenStream>::new();
        let mut offset = 0u32;
        for field in &fields {
            let id = field.constant_id;
            let entry = quote::quote! {
                ::fzvk::vk::SpecializationMapEntry {
                    constant_id: #id,
                    offset: #offset,
                    size: 4,
                }
            };
            specialization_entries.push(entry);
            offset += 4;
        }
        // Crimes to get quote repetitions to work
        let field_name_1 = fields.iter().map(|field| &field.name);
        let field_name_2 = field_name_1.clone();
        let field_ty = fields.iter().map(|field| &field.ty);
        let field_default_value = fields.iter().map(|field| &field.default_value);
        // Definition and Specialization trait implementation
        let def = quote::quote! {
            /// Specialization constants, generated from SPIR-V.
            ///
            /// The `Default::default` implementation takes the default values
            /// from the shader.
            #[repr(C)]
            pub struct #name {
                #(pub #field_name_1: #field_ty),*
            }
            impl ::core::default::Default for #name {
                fn default() -> Self {
                    Self {
                        #(#field_name_2: #field_default_value),*
                    }
                }
            }
            unsafe impl ::fzvk::pipeline::Specialization for #name {
                const ENTRIES : &[::fzvk::vk::SpecializationMapEntry] = &[#(#specialization_entries),*];
            }
        };
        // Is it clear I don't know what im doing >~<
        (def, quote::quote! {#name})
    };
    // Generate constants representing the entry points.
    let entry_constants = {
        struct EntryConst {
            const_name: proc_macro2::Ident,
            entry_point: proc_macro2::Literal,
            ty_constant: proc_macro2::TokenStream,
        }
        let mut entries = Vec::<EntryConst>::new();
        for this_entry in &module.entry_points {
            use convert_case::Casing;

            let is_unique = module
                .entry_points
                .iter()
                .filter(|entry| entry.name == this_entry.name)
                .count()
                == 1;
            let entry_name = std::ffi::CString::new(this_entry.name.clone()).unwrap();
            let upper_snake_case = this_entry.name.to_case(convert_case::Case::UpperSnake);
            let const_name = if !is_unique {
                // Append the execution model, resulting in e.g. `MAIN_COMPUTE`.
                // *this* is guaranteed unique, as there can't be several entry
                // points with the same name *and* execution model in a SPIRV
                // module.
                upper_snake_case + "_" + module::execution_model_upper_str(this_entry.model)
            } else {
                upper_snake_case
            };
            let model_typestate_marker = module::execution_model_typestate(this_entry.model);
            entries.push(EntryConst {
                const_name: quote::format_ident!("{const_name}"),
                entry_point: proc_macro2::Literal::c_string(&entry_name),
                ty_constant: model_typestate_marker.parse().unwrap(),
            });
        }
        let const_name = entries.iter().map(|entry| &entry.const_name);
        let entry_point = entries.iter().map(|entry| &entry.entry_point);
        let ty_constant = entries.iter().map(|entry| &entry.ty_constant);
        // Assemble the constant!
        quote::quote! {
            #(
                pub const #const_name :
                    ::fzvk::pipeline::StaticEntryName::<'static, Module, #ty_constant>
                    = unsafe {
                        ::fzvk::pipeline::StaticEntryName::new(#entry_point)
                    };
            )*
        }
    };

    let module_version = {
        let (major, minor) = module.version;
        quote::quote! {(#major, #minor)}
    };

    // Finally, put it all together! Along with the trait definition which holds
    // the &[u32] SPIR-V bytecode.
    quote::quote! {
        #specialization_struct_def
        pub struct Module;
        unsafe impl ::fzvk::pipeline::StaticSpirV for Module {
            type Specialization = #specialization_struct_name;
            const SPIRV : &[u32] = &[#(#spirv),*];
            const VERSION : (u8, u8) = #module_version;
        }
        #entry_constants
    }
    .into()
}
