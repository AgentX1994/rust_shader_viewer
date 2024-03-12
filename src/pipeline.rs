use log::info;

use crate::shader::Shader;

pub struct PipelineCreateInfo<'a> {
    pub color_format: wgpu::TextureFormat,
    pub depth_format: Option<wgpu::TextureFormat>,
    pub vertex_layouts: &'a [wgpu::VertexBufferLayout<'a>],
    pub topology: wgpu::PrimitiveTopology,
    pub shader: &'a Shader,
    pub label: Option<&'a str>,
}

pub struct RenderPipeline {
    pipeline_layout: wgpu::PipelineLayout,
    pipeline: wgpu::RenderPipeline,
}

impl RenderPipeline {
    fn create_pipeline(
        device: &wgpu::Device,
        layout: &wgpu::PipelineLayout,
        create_info: PipelineCreateInfo,
    ) -> wgpu::RenderPipeline {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: create_info.label,
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: create_info.shader.get_vertex_module(),
                entry_point: create_info.shader.get_vertex_entry_point(),
                buffers: create_info.vertex_layouts,
            },
            fragment: Some(wgpu::FragmentState {
                module: create_info.shader.get_fragment_module(),
                entry_point: create_info.shader.get_fragment_entry_point(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: create_info.color_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: create_info.topology,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: create_info
                .depth_format
                .map(|format| wgpu::DepthStencilState {
                    format,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        })
    }
    pub fn new(
        device: &wgpu::Device,
        layout: wgpu::PipelineLayout,
        create_info: PipelineCreateInfo,
    ) -> Self {
        let name = create_info.shader.name();
        let pipeline = Self::create_pipeline(device, &layout, create_info);
        info!("Pipeline created for shader {}", name);
        Self {
            pipeline_layout: layout,
            pipeline,
        }
    }

    pub fn recreate(&mut self, device: &wgpu::Device, create_info: PipelineCreateInfo) {
        let name = create_info.shader.name();
        self.pipeline = Self::create_pipeline(device, &self.pipeline_layout, create_info);
        info!("Pipeline recreated for shader {}", name);
    }

    pub fn pipeline(&self) -> &wgpu::RenderPipeline {
        &self.pipeline
    }
}
