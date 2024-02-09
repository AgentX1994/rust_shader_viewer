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
        source: wgpu::ShaderSource,
    ) -> Self {
        let desc = wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source,
        };
        let module = ShaderModule::Wgsl {
            module: device.create_shader_module(desc),
        };
        Self {
            name: name.into(),
            vertex_entry_point: vertex_entry_point.to_owned(),
            fragment_entry_point: fragment_entry_point.to_owned(),
            module,
        }
    }
    pub fn new_wgsl(
        device: &wgpu::Device,
        name: &str,
        vertex_entry_point: &str,
        fragment_entry_point: &str,
        source: &str,
    ) -> Self {
        Self::new(
            device,
            name,
            vertex_entry_point,
            fragment_entry_point,
            wgpu::ShaderSource::Wgsl(source.into()),
        )
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
