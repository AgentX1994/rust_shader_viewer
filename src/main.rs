use rust_shader_viewer::run;

pub fn main() {
    env_logger::init();
    match pollster::block_on(run()) {
        Ok(_) => (),
        Err(e) => eprintln!("{}", e),
    }
}
