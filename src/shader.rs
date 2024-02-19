use log::{error, warn};

use naga::front::{glsl, wgsl};

use crate::error::{RendererError, RendererResult};

fn get_entry_points(name: &str, modules: &[&naga::Module]) -> RendererResult<(String, String)> {
    let mut vertex_entry_point: Option<String> = None;
    let mut fragment_entry_point: Option<String> = None;
    for module in modules {
        for entry_point in &module.entry_points {
            match entry_point.stage {
                naga::ShaderStage::Vertex => {
                    if vertex_entry_point.is_some() {
                        warn!("Shader {} has more than one vertex entry point!", name);
                    } else {
                        vertex_entry_point = Some(entry_point.name.clone());
                    }
                }
                naga::ShaderStage::Fragment => {
                    if fragment_entry_point.is_some() {
                        warn!("Shader {} has more than one fragment entry point!", name);
                    } else {
                        fragment_entry_point = Some(entry_point.name.clone());
                    }
                }
                naga::ShaderStage::Compute => (),
            }
        }
    }

    match (vertex_entry_point, fragment_entry_point) {
        (None, None) => {
            error!("Shader {} has no vertex or fragment entry points!", name);
            Err(RendererError::ShaderCompile(
                "Shader has no entry points!".into(),
            ))
        }
        (None, Some(_)) => {
            error!("Shader {} has no vertex entry point!", name);
            Err(RendererError::ShaderCompile(
                "Shader has no vertex entry point!".into(),
            ))
        }
        (Some(_), None) => {
            error!("Shader {} has no fragment entry point!", name);
            Err(RendererError::ShaderCompile(
                "Shader has no fragment entry point!".into(),
            ))
        }
        (Some(v), Some(s)) => Ok((v, s)),
    }
}

enum ShaderInput<'a> {
    Wgsl(wgpu::ShaderSource<'a>),
    Glsl {
        vertex: wgpu::ShaderSource<'a>,
        fragment: wgpu::ShaderSource<'a>,
    },
}

enum ShaderModule {
    Wgsl {
        module: wgpu::ShaderModule,
    },
    Glsl {
        vertex: wgpu::ShaderModule,
        fragment: wgpu::ShaderModule,
    },
}

pub struct Shader {
    name: String,
    vertex_entry_point: String,
    fragment_entry_point: String,
    module: ShaderModule,
}

impl Shader {
    fn new(
        device: &wgpu::Device,
        name: &str,
        vertex_entry_point: &str,
        fragment_entry_point: &str,
        source: ShaderInput,
    ) -> Self {
        let module = match source {
            ShaderInput::Wgsl(source) => {
                let desc = wgpu::ShaderModuleDescriptor {
                    label: Some(name),
                    source,
                };
                ShaderModule::Wgsl {
                    module: device.create_shader_module(desc),
                }
            }
            ShaderInput::Glsl { vertex, fragment } => {
                let vert_desc = wgpu::ShaderModuleDescriptor {
                    label: Some(name),
                    source: vertex,
                };
                let frag_desc = wgpu::ShaderModuleDescriptor {
                    label: Some(name),
                    source: fragment,
                };
                ShaderModule::Glsl {
                    vertex: device.create_shader_module(vert_desc),
                    fragment: device.create_shader_module(frag_desc),
                }
            }
        };
        Self {
            name: name.into(),
            vertex_entry_point: vertex_entry_point.to_owned(),
            fragment_entry_point: fragment_entry_point.to_owned(),
            module,
        }
    }

    pub fn new_wgsl(device: &wgpu::Device, name: &str, source: &str) -> RendererResult<Self> {
        let mut frontend = wgsl::Frontend::new();
        let module = frontend.parse(source)?;
        let (vertex_entry_point, fragment_entry_point) = get_entry_points(name, &[&module])?;
        Ok(Self::new(
            device,
            name,
            &vertex_entry_point,
            &fragment_entry_point,
            ShaderInput::Wgsl(wgpu::ShaderSource::Wgsl(source.into())),
        ))
    }

    fn new_glsl(
        device: &wgpu::Device,
        name: &str,
        vertex_source: &str,
        fragment_source: &str,
    ) -> RendererResult<Self> {
        let mut frontend = glsl::Frontend::default();
        let vert_module = frontend.parse(
            &glsl::Options::from(naga::ShaderStage::Vertex),
            vertex_source,
        )?;
        let frag_module = frontend.parse(
            &glsl::Options::from(naga::ShaderStage::Vertex),
            fragment_source,
        )?;
        let (vertex_entry_point, fragment_entry_point) =
            get_entry_points(name, &[&vert_module, &frag_module])?;
        Ok(Self::new(
            device,
            name,
            &vertex_entry_point,
            &fragment_entry_point,
            ShaderInput::Glsl {
                vertex: wgpu::ShaderSource::Glsl {
                    shader: vertex_source.into(),
                    stage: naga::ShaderStage::Vertex,
                    defines: Default::default(),
                },
                fragment: wgpu::ShaderSource::Glsl {
                    shader: fragment_source.into(),
                    stage: naga::ShaderStage::Fragment,
                    defines: Default::default(),
                },
            },
        ))
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn get_vertex_module(&self) -> &wgpu::ShaderModule {
        match &self.module {
            ShaderModule::Wgsl { module } => module,
            ShaderModule::Glsl { vertex, .. } => vertex,
        }
    }

    pub fn get_vertex_entry_point(&self) -> &str {
        &self.vertex_entry_point
    }

    pub fn get_fragment_module(&self) -> &wgpu::ShaderModule {
        match &self.module {
            ShaderModule::Wgsl { module } => module,
            ShaderModule::Glsl { fragment, .. } => fragment,
        }
    }

    pub fn get_fragment_entry_point(&self) -> &str {
        &self.fragment_entry_point
    }
}
