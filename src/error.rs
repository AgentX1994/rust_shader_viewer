use std::error;

use thiserror::Error;
use wgpu::naga::front::glsl;
use wgpu::naga::front::wgsl;

use image::ImageError;
use tobj::LoadError;
use winit::error::EventLoopError;

#[derive(Debug, Error)]
pub enum RendererError {
    #[error("Error Decoding HDR image: {0}")]
    HdrDecoding(#[from] ImageError),
    #[error("Error reading file: {0}")]
    Io(#[from] std::io::Error),
    #[error("Error parsing WGSL source: {0}")]
    WgslShaderParse(#[from] wgsl::ParseError),
    #[error("Error parsing GLSL source: {0:?}")]
    GlslShaderParse(Vec<glsl::Error>),
    #[error("Error compiling shader: {0}")]
    ShaderCompile(String),
    #[error("Error loading model: {0}")]
    ObjLoad(#[from] LoadError),
    #[error("Error running event loop: {0}")]
    EventLoop(#[from] EventLoopError),
    #[error("WGPU Validation Error: {source}\n\tDescription: {description}")]
    WgpuValidationError {
        #[source]
        source: Box<dyn error::Error + Send + 'static>,
        description: String,
    },
}

impl From<Vec<glsl::Error>> for RendererError {
    fn from(value: Vec<glsl::Error>) -> Self {
        RendererError::GlslShaderParse(value)
    }
}

pub type RendererResult<T> = Result<T, RendererError>;
