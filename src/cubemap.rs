use crate::error::RendererResult;
use crate::pipeline::RenderPipeline;
use crate::resources;
use crate::shader::Shader;
use crate::texture::{CubeTexture, Texture};

pub struct CubeMapRenderer {
    label: Option<String>,
    _texture: CubeTexture,
    layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    pipeline: RenderPipeline,
}

impl CubeMapRenderer {
    pub async fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        camera_bind_group_layout_desc: &wgpu::BindGroupLayoutDescriptor<'_>,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
        surface_format: wgpu::TextureFormat,
        filename: &str,
        label: Option<&str>,
    ) -> RendererResult<Self> {
        // TODO error handling
        let texture = {
            let hdr_loader = resources::HdrLoader::new(device);
            let sky_bytes = resources::load_binary(filename).await?;
            hdr_loader.cube_from_equirectangular_bytes(device, queue, &sky_bytes, 1080, label)?
        };

        let desc = wgpu::BindGroupLayoutDescriptor {
            label,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::Cube,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        };
        let layout = device.create_bind_group_layout(&desc);

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label,
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(texture.view()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(texture.sampler()),
                },
            ],
        });

        let pipeline = {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label,
                bind_group_layouts: &[camera_bind_group_layout, &layout],
                push_constant_ranges: &[],
            });

            let shader = Shader::new_wgsl(
                device,
                "cubemap",
                include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/shaders/sky.wgsl")),
            )
            .expect("Could not parse cubemap shader");
            assert!(shader.layout_matches(&[camera_bind_group_layout_desc, &desc]));
            RenderPipeline::new(
                device,
                &layout,
                surface_format,
                Some(Texture::DEPTH_FORMAT),
                &[],
                wgpu::PrimitiveTopology::TriangleList,
                &shader,
            )
        };

        Ok(Self {
            label: label.map(str::to_owned),
            _texture: texture,
            layout,
            bind_group,
            pipeline,
        })
    }

    pub fn layout(&self) -> &wgpu::BindGroupLayout {
        &self.layout
    }

    pub fn layout_desc(&self) -> wgpu::BindGroupLayoutDescriptor {
        wgpu::BindGroupLayoutDescriptor {
            label: self.label.as_deref(),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::Cube,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        }
    }

    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    pub fn render<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        camera_bind_group: &'a wgpu::BindGroup,
    ) {
        render_pass.set_pipeline(&self.pipeline.pipeline);
        render_pass.set_bind_group(0, camera_bind_group, &[]);
        render_pass.set_bind_group(1, &self.bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }
}
