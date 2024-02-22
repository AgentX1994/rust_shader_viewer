use std::borrow::Cow;

use log::{error, info, warn};

use wgpu::naga::front::{glsl, wgsl};

use crate::error::{RendererError, RendererResult};

fn get_entry_points(
    name: &str,
    modules: &[&wgpu::naga::Module],
) -> RendererResult<(String, String)> {
    let mut vertex_entry_point: Option<String> = None;
    let mut fragment_entry_point: Option<String> = None;
    for module in modules {
        for entry_point in &module.entry_points {
            match entry_point.stage {
                wgpu::naga::ShaderStage::Vertex => {
                    if vertex_entry_point.is_some() {
                        warn!("Shader {} has more than one vertex entry point!", name);
                    } else {
                        vertex_entry_point = Some(entry_point.name.clone());
                    }
                }
                wgpu::naga::ShaderStage::Fragment => {
                    if fragment_entry_point.is_some() {
                        warn!("Shader {} has more than one fragment entry point!", name);
                    } else {
                        fragment_entry_point = Some(entry_point.name.clone());
                    }
                }
                wgpu::naga::ShaderStage::Compute => (),
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

#[derive(Debug, Default)]
struct OwningBindGroupLayoutDescriptor {
    label: Option<String>,
    entries: Vec<wgpu::BindGroupLayoutEntry>,
}

fn naga_type_to_binding_group_type(ty: &wgpu::naga::Type) -> wgpu::BindingType {
    use wgpu::naga::TypeInner;
    match ty.inner {
        TypeInner::Scalar(_)
        | TypeInner::Vector { .. }
        | TypeInner::Matrix { .. }
        | TypeInner::Atomic(_)
        | TypeInner::Array { .. }
        | TypeInner::Struct { .. } => wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        TypeInner::Image {
            dim,
            arrayed: _,
            class,
        } => {
            use wgpu::naga::{ImageClass, ImageDimension, ScalarKind};
            use wgpu::{TextureSampleType, TextureViewDimension};
            let (sample_type, multisampled) = match class {
                ImageClass::Sampled { kind, multi } => match kind {
                    ScalarKind::Sint => (TextureSampleType::Sint, multi),
                    ScalarKind::Uint => (TextureSampleType::Uint, multi),
                    ScalarKind::Float => (TextureSampleType::Float { filterable: true }, multi),
                    ScalarKind::Bool => todo!("What's the texture sample type for bools?"),
                    ScalarKind::AbstractInt => unreachable!(),
                    ScalarKind::AbstractFloat => unreachable!(),
                },
                ImageClass::Depth { multi } => (TextureSampleType::Depth, multi),
                ImageClass::Storage {
                    format: _,
                    access: _,
                } => {
                    todo!("How to deal with storage images")
                }
            };
            let view_dimension = match dim {
                ImageDimension::D1 => TextureViewDimension::D1,
                ImageDimension::D2 => TextureViewDimension::D2,
                ImageDimension::D3 => TextureViewDimension::D3,
                ImageDimension::Cube => TextureViewDimension::Cube,
            };
            wgpu::BindingType::Texture {
                sample_type,
                view_dimension,
                multisampled,
            }
        }
        TypeInner::Sampler { comparison } => {
            if comparison {
                wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison)
            } else {
                wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)
            }
        }
        _ => todo!("Unimplemented Type for uniform: {:?}", ty.inner),
    }
}

fn get_binding_layout(
    modules: &[(&str, wgpu::ShaderStages, &wgpu::naga::Module)],
) -> Vec<OwningBindGroupLayoutDescriptor> {
    let mut layouts = Vec::new();
    for (name, shader_type, module) in modules {
        for (_handle, global) in module.global_variables.iter() {
            let typ = &module.types[global.ty];
            let (group, binding) = global
                .binding
                .as_ref()
                .map(|binding| (binding.group, binding.binding))
                .unzip();
            info!(
                "Module {} ({:?} shader) has global {} (group {}, binding {}) has type {:?}",
                name,
                shader_type,
                global.name.as_deref().unwrap_or("<Unnamed>"),
                group.map(|g| g.to_string()).unwrap_or("None".to_string()),
                binding.map(|g| g.to_string()).unwrap_or("None".to_string()),
                typ
            );
            if let Some(binding) = &global.binding {
                if layouts.len() < binding.group as usize + 1 {
                    layouts.resize_with(
                        binding.group as usize + 1,
                        OwningBindGroupLayoutDescriptor::default,
                    );
                }
                let entry = &mut layouts[binding.group as usize];
                if let Some(name) = &global.name {
                    let s = entry.label.get_or_insert_with(String::default);
                    if !s.is_empty() {
                        s.push(',');
                    }
                    s.push_str(name);
                }
                entry.entries.push(wgpu::BindGroupLayoutEntry {
                    binding: binding.binding,
                    visibility: *shader_type,
                    ty: naga_type_to_binding_group_type(typ),
                    count: None,
                })
            } else {
                info!("Not a uniform, skipping for now.");
            }
        }
    }
    layouts
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
    layout: Vec<OwningBindGroupLayoutDescriptor>,
}

impl Shader {
    fn new(
        device: &wgpu::Device,
        name: &str,
        vertex_entry_point: &str,
        fragment_entry_point: &str,
        source: ShaderInput,
        layout: Vec<OwningBindGroupLayoutDescriptor>,
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
            layout,
        }
    }

    pub fn new_wgsl(device: &wgpu::Device, name: &str, source: &str) -> RendererResult<Self> {
        let mut frontend = wgsl::Frontend::new();
        let module = frontend.parse(source)?;
        let (vertex_entry_point, fragment_entry_point) = get_entry_points(name, &[&module])?;
        let layout = get_binding_layout(&[(name, wgpu::ShaderStages::VERTEX_FRAGMENT, &module)]);
        info!("Layout for shader {}: {:?}", name, layout);
        Ok(Self::new(
            device,
            name,
            &vertex_entry_point,
            &fragment_entry_point,
            ShaderInput::Wgsl(wgpu::ShaderSource::Naga(Cow::Owned(module))),
            layout,
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
            &glsl::Options::from(wgpu::naga::ShaderStage::Vertex),
            vertex_source,
        )?;
        let frag_module = frontend.parse(
            &glsl::Options::from(wgpu::naga::ShaderStage::Vertex),
            fragment_source,
        )?;
        let (vertex_entry_point, fragment_entry_point) =
            get_entry_points(name, &[&vert_module, &frag_module])?;
        let layout = get_binding_layout(&[
            (name, wgpu::ShaderStages::VERTEX, &vert_module),
            (name, wgpu::ShaderStages::FRAGMENT, &frag_module),
        ]);
        info!("Layout for shader {}: {:?}", name, layout);
        Ok(Self::new(
            device,
            name,
            &vertex_entry_point,
            &fragment_entry_point,
            ShaderInput::Glsl {
                vertex: wgpu::ShaderSource::Naga(Cow::Owned(vert_module)),
                fragment: wgpu::ShaderSource::Naga(Cow::Owned(frag_module)),
            },
            layout,
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

    pub fn get_layout(&self) -> Vec<wgpu::BindGroupLayoutDescriptor> {
        self.layout
            .iter()
            .map(|d| wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &d.entries,
            })
            .collect()
    }
}
