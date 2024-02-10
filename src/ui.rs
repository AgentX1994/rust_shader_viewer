use egui::Context;
use egui_wgpu::{Renderer, ScreenDescriptor};
use egui_winit::State;
use winit::{event::WindowEvent, window::Window};

pub struct EguiDrawParams<'a, U: FnOnce(&Context)> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub encoder: &'a mut wgpu::CommandEncoder,
    pub window: &'a Window,
    pub view: &'a wgpu::TextureView,
    pub screen_descriptor: ScreenDescriptor,
    pub run_ui: U,
}

pub struct EguiRenderer {
    state: State,
    renderer: Renderer,
}

impl EguiRenderer {
    pub fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        depth_format: Option<wgpu::TextureFormat>,
        msaa_samples: u32,
        window: &Window,
    ) -> Self {
        let context = Context::default();
        let viewport_id = context.viewport_id();
        let state = State::new(context, viewport_id, &window, None, None);
        let renderer = Renderer::new(device, format, depth_format, msaa_samples);

        EguiRenderer { state, renderer }
    }

    pub fn handle_input(&mut self, window: &Window, event: &WindowEvent) -> bool {
        let res = self.state.on_window_event(window, event);
        res.consumed
    }

    pub fn draw<U: FnOnce(&Context)>(&mut self, draw_params: EguiDrawParams<'_, U>) {
        let EguiDrawParams {
            device,
            queue,
            encoder,
            window,
            view,
            screen_descriptor,
            run_ui,
        } = draw_params;
        let raw_input = self.state.take_egui_input(window);
        let full_output = self.state.egui_ctx().run(raw_input, |ui| run_ui(ui));

        self.state
            .handle_platform_output(window, full_output.platform_output);
        let tris = self
            .state
            .egui_ctx()
            .tessellate(full_output.shapes, window.scale_factor() as f32);
        for (id, image_delta) in &full_output.textures_delta.set {
            self.renderer
                .update_texture(device, queue, *id, image_delta);
        }
        self.renderer
            .update_buffers(device, queue, encoder, &tris, &screen_descriptor);
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            label: Some("egui main render pass"),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        self.renderer.render(&mut rpass, &tris, &screen_descriptor);
        drop(rpass);
        for x in &full_output.textures_delta.free {
            self.renderer.free_texture(x);
        }
    }
}
