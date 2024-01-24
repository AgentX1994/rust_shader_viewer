use rust_shader_viewer::run;

pub fn main() {
    env_logger::init();
    pollster::block_on(run());
}
