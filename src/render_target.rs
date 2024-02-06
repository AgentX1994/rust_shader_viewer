use crate::surface::Surface;

pub trait RenderTarget {
    fn view(&self) -> &wgpu::TextureView;
}

pub struct SurfaceTextureRenderTarget {
    image: wgpu::SurfaceTexture,
    view: wgpu::TextureView,
}

impl SurfaceTextureRenderTarget {
    pub fn new(surface: &Surface) -> Result<Self, wgpu::SurfaceError> {
        let image = surface.get_current_texture()?;
        let view = image
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        Ok(Self { image, view })
    }

    pub fn present(self) {
        self.image.present()
    }
}

impl RenderTarget for SurfaceTextureRenderTarget {
    fn view(&self) -> &wgpu::TextureView {
        &self.view
    }
}
