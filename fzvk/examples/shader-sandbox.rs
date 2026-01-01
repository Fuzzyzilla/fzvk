const _: () = {
    fzvk_shader::glsl! {
        file: "shader.glsl",
        defines: {
            should_be_defined,
            should_be_true: true,
            "should_be_five": 5,
        }
    }
};

fn main() {}
