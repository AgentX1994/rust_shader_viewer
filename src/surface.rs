pub struct Surface {
    surface: wgpu::Surface<'static>,
    capabilities: wgpu::SurfaceCapabilities,
    config: wgpu::SurfaceConfiguration,
}

impl Surface {
    pub fn new(
        size: (u32, u32),
        surface: wgpu::Surface<'static>,
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
    ) -> Self {
        let capabilities = surface.get_capabilities(adapter);
        let surface_format = *capabilities
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .unwrap_or(&capabilities.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            format: surface_format,
            ..surface
                .get_default_config(adapter, size.0, size.1)
                .expect("No default config!")
        };
        surface.configure(device, &config);
        Self {
            surface,
            capabilities,
            config,
        }
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(device, &self.config);
    }

    pub fn extent(&self) -> wgpu::Extent3d {
        wgpu::Extent3d {
            width: self.config.width,
            height: self.config.height,
            depth_or_array_layers: 1,
        }
    }

    pub fn surface(&self) -> &wgpu::Surface {
        &self.surface
    }

    pub fn capabilities(&self) -> &wgpu::SurfaceCapabilities {
        &self.capabilities
    }

    pub fn format(&self) -> wgpu::TextureFormat {
        self.config.format
    }

    pub fn get_current_texture(&self) -> Result<wgpu::SurfaceTexture, wgpu::SurfaceError> {
        self.surface.get_current_texture()
    }
}
