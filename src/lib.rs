use std::{
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

use egui::{Color32, RichText};
use egui_wgpu::ScreenDescriptor;
use error::RendererResult;
use winit::{
    event::*,
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowBuilder},
};

use cgmath::prelude::*;
use log::{error, info};
use wgpu::util::DeviceExt;

mod camera;
mod cubemap;
mod error;
mod hdr;
mod light;
mod model;
mod pipeline;
mod render_target;
mod resources;
mod shader;
mod surface;
mod texture;
mod ui;

use camera::{Camera, CameraController, PerspectiveCamera, Projection};
use cubemap::CubeMapRenderer;
use light::LightUniform;
use model::{Instance, LightRenderer, ModelRenderer, Vertex};
use pipeline::RenderPipeline;
use render_target::{RenderTarget, SurfaceTextureRenderTarget};
use shader::Shader;
use surface::Surface;
use ui::{highlight, CodeTheme, EguiDrawParams, EguiRenderer};

const fn get_texture_layout_desc() -> wgpu::BindGroupLayoutDescriptor<'static> {
    const LAYOUT_ENTRIES: [wgpu::BindGroupLayoutEntry; 4] = [
        // diffuse texture
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                multisampled: false,
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        },
        // normal map
        wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                multisampled: false,
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 3,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        },
    ];
    wgpu::BindGroupLayoutDescriptor {
        label: Some("Bind Group Layout"),
        entries: &LAYOUT_ENTRIES,
    }
}

const fn get_light_layout_desc() -> wgpu::BindGroupLayoutDescriptor<'static> {
    const LAYOUT_ENTRIES: [wgpu::BindGroupLayoutEntry; 1] = [wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }];
    wgpu::BindGroupLayoutDescriptor {
        label: Some("Light Bind Group Layout"),
        entries: &LAYOUT_ENTRIES,
    }
}

struct State {
    surface: Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    size: winit::dpi::PhysicalSize<u32>,
    camera: PerspectiveCamera,
    camera_controller: CameraController,
    mouse_pressed: bool,
    depth_texture: texture::Texture,
    shader_source: String,
    shader_compile_error: Option<String>,
    render_pipeline_layout: wgpu::PipelineLayout,
    render_pipeline: RenderPipeline,
    light_render_pipeline: RenderPipeline,
    model: model::Model,
    instances: Vec<Instance>,
    instance_buffer: wgpu::Buffer,
    light: LightUniform,
    light_buffer: wgpu::Buffer,
    light_bind_group: wgpu::BindGroup,
    hdr: hdr::HdrPipeline,
    cubemap: CubeMapRenderer,
    ui: EguiRenderer,
}

impl State {
    async fn new<P: AsRef<Path>>(
        window: Arc<Window>,
        starting_shader: Option<P>,
    ) -> RendererResult<Self> {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::all_webgpu_mask(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .unwrap();

        let surface = Surface::new((size.width, size.height), surface, &adapter, &device);
        info!(
            "Surface created, surface: {:?}, capabilities: {:?}",
            surface.surface(),
            surface.capabilities()
        );
        let surface_size = surface.extent();

        let depth_texture =
            texture::Texture::create_depth_texture(&device, &surface_size, "depth_texture");

        let texture_bind_group_layout_desc = get_texture_layout_desc();
        let texture_bind_group_layout =
            device.create_bind_group_layout(&texture_bind_group_layout_desc);

        let camera = Camera::new((0.0, 5.0, 10.0), cgmath::Deg(-90.0), cgmath::Deg(-20.0));
        let projection = Projection::new(
            surface_size.width,
            surface_size.height,
            cgmath::Deg(45.0),
            0.1,
            100.0,
        );

        let camera = PerspectiveCamera::new(&device, camera, projection);

        // Create instances
        const NUM_INSTANCE_PER_ROW: usize = 10;
        let model = resources::load_model("cube.obj", &device, &queue, &texture_bind_group_layout)
            .await
            .expect("Failed to load model!");

        const SPACE_BETWEEN: f32 = 3.0;
        let instances = (0..NUM_INSTANCE_PER_ROW)
            .flat_map(|z| {
                (0..NUM_INSTANCE_PER_ROW).map(move |x| {
                    let x = SPACE_BETWEEN * (x as f32 - NUM_INSTANCE_PER_ROW as f32 / 2.0);
                    let z = SPACE_BETWEEN * (z as f32 - NUM_INSTANCE_PER_ROW as f32 / 2.0);
                    let position = cgmath::Vector3 { x, y: 0.0, z };
                    let rotation = if position.is_zero() {
                        cgmath::Quaternion::from_axis_angle(
                            cgmath::Vector3::unit_z(),
                            cgmath::Deg(0.0),
                        )
                    } else {
                        cgmath::Quaternion::from_axis_angle(position.normalize(), cgmath::Deg(45.0))
                    };

                    Instance {
                        position,
                        rotation,
                        rotation_speed: 0.025f32,
                        rotation_axis: cgmath::Vector3::unit_y(),
                    }
                })
            })
            .collect::<Vec<Instance>>();
        let instance_data = instances.iter().map(Instance::to_raw).collect::<Vec<_>>();

        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let light = LightUniform::new([2.0, 2.0, 2.0], [1.0, 1.0, 1.0]);

        let light_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Light Buffer"),
            contents: bytemuck::cast_slice(&[light]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let light_bind_group_layout_desc = get_light_layout_desc();
        let light_bind_group_layout =
            device.create_bind_group_layout(&light_bind_group_layout_desc);

        let light_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Light Bind Group"),
            layout: &light_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: light_buffer.as_entire_binding(),
            }],
        });

        let hdr = hdr::HdrPipeline::new(&device, &surface_size, surface.format());
        let cubemap = CubeMapRenderer::new(
            &device,
            &queue,
            &PerspectiveCamera::layout_desc(),
            camera.layout(),
            hdr.format(),
            "pure-sky.hdr",
            Some("Sky Cubemap"),
        )
        .await?;

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Pipeline layout"),
                bind_group_layouts: &[
                    &texture_bind_group_layout,
                    camera.layout(),
                    &light_bind_group_layout,
                    cubemap.layout(),
                ],
                push_constant_ranges: &[],
            });

        let shader_source = starting_shader
            .and_then(|p| match std::fs::read_to_string(p.as_ref()) {
                Ok(s) => Some(s),
                Err(e) => {
                    error!(
                        "Unable to load supplied shader {} ({}), using default!",
                        p.as_ref().display(),
                        e
                    );
                    None
                }
            })
            .unwrap_or(
                include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/shaders/shader.wgsl"))
                    .to_string(),
            );

        let render_pipeline = {
            let shader = Shader::new_wgsl(&device, "normal", &shader_source)
                .expect("Could not parse normal shader");
            assert!(shader.layout_matches(&[
                &texture_bind_group_layout_desc,
                &PerspectiveCamera::layout_desc(),
                &light_bind_group_layout_desc,
                &cubemap.layout_desc()
            ]));
            RenderPipeline::new(
                &device,
                &render_pipeline_layout,
                hdr.format(),
                Some(texture::Texture::DEPTH_FORMAT),
                &[model::ModelVertex::layout(), Instance::desc()],
                wgpu::PrimitiveTopology::TriangleList,
                &shader,
            )
        };

        let light_render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Pipeline layout"),
                bind_group_layouts: &[camera.layout(), &light_bind_group_layout],
                push_constant_ranges: &[],
            });

        let light_render_pipeline = {
            let shader = Shader::new_wgsl(
                &device,
                "Light",
                include_str!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/shaders/light_shader.wgsl"
                )),
            )
            .expect("Could not parse light shader");
            assert!(shader.layout_matches(&[
                &PerspectiveCamera::layout_desc(),
                &light_bind_group_layout_desc
            ]));
            RenderPipeline::new(
                &device,
                &light_render_pipeline_layout,
                hdr.format(),
                Some(texture::Texture::DEPTH_FORMAT),
                &[model::ModelVertex::layout()],
                wgpu::PrimitiveTopology::TriangleList,
                &shader,
            )
        };
        let ui = EguiRenderer::new(&device, hdr.format(), None, 1, &window);
        Ok(Self {
            surface,
            device,
            queue,
            size,
            camera,
            camera_controller: CameraController::new(4.0, 1.0),
            mouse_pressed: false,
            depth_texture,
            render_pipeline_layout,
            render_pipeline,
            shader_source,
            shader_compile_error: None,
            light_render_pipeline,
            model,
            instances,
            instance_buffer,
            light,
            light_buffer,
            light_bind_group,
            hdr,
            cubemap,
            ui,
        })
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            info!("Resizing to {}x{}", new_size.width, new_size.height);
            self.size = new_size;
            self.surface
                .resize(&self.device, new_size.width, new_size.height);
            self.camera.resize(new_size.width, new_size.height);
            self.depth_texture = texture::Texture::create_depth_texture(
                &self.device,
                &self.surface.extent(),
                "depth_texture",
            );
            self.hdr
                .resize(&self.device, new_size.width, new_size.height);
        }
    }

    fn input(&mut self, window: &Window, event: &WindowEvent) -> bool {
        if self.ui.handle_input(window, event) {
            return true;
        }
        match event {
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(key),
                        state,
                        ..
                    },
                ..
            } => self.camera_controller.process_keyboard(*key, *state),
            WindowEvent::MouseWheel { delta, .. } => {
                self.camera_controller.process_scroll(delta);
                true
            }
            WindowEvent::MouseInput {
                button: MouseButton::Left,
                state,
                ..
            } => {
                self.mouse_pressed = *state == ElementState::Pressed;
                true
            }
            _ => false,
        }
    }

    fn process_mouse_motion(&mut self, mouse_dx: f64, mouse_dy: f64) {
        if self.mouse_pressed {
            self.camera_controller.process_mouse(mouse_dx, mouse_dy);
        }
    }

    fn compile_shader(&mut self) {
        info!("Compiling shader");
        let shader = match Shader::new_wgsl(&self.device, "shader", &self.shader_source) {
            Ok(s) => {
                self.shader_compile_error = None;
                s
            }
            Err(e) => {
                self.shader_compile_error = Some(e.to_string());
                return;
            }
        };
        if !shader.layout_matches(&[
            &get_texture_layout_desc(),
            &PerspectiveCamera::layout_desc(),
            &get_light_layout_desc(),
            &self.cubemap.layout_desc(),
        ]) {
            self.shader_compile_error = Some("Shader layout doesn't match!".to_string());
            return;
        }
        self.render_pipeline = RenderPipeline::new(
            &self.device,
            &self.render_pipeline_layout,
            self.hdr.format(),
            Some(texture::Texture::DEPTH_FORMAT),
            &[model::ModelVertex::layout(), Instance::desc()],
            wgpu::PrimitiveTopology::TriangleList,
            &shader,
        );
    }

    fn load_shader<P: AsRef<Path>>(&mut self, shader_file_path: P) -> RendererResult<()> {
        let source = std::fs::read_to_string(shader_file_path)?;
        self.shader_source = source;
        self.compile_shader();
        Ok(())
    }

    fn update(&mut self, dt: Duration) {
        // TODO This is a clumsy way to update the camera
        self.camera
            .update(&mut self.camera_controller, dt, &self.queue);

        for ele in &mut self.instances {
            ele.update();
        }
        let instance_data = self
            .instances
            .iter()
            .map(Instance::to_raw)
            .collect::<Vec<_>>();
        self.queue.write_buffer(
            &self.instance_buffer,
            0,
            bytemuck::cast_slice(&instance_data),
        );

        // Update light
        let old_position: cgmath::Vector3<_> = self.light.position.into();
        self.light.position = (cgmath::Quaternion::from_axis_angle(
            cgmath::Vector3::unit_y(),
            cgmath::Deg(60.0 * dt.as_secs_f32()),
        ) * old_position)
            .into();
        self.queue
            .write_buffer(&self.light_buffer, 0, bytemuck::cast_slice(&[self.light]));
    }

    fn render(&mut self, window: &Window) -> Result<(), wgpu::SurfaceError> {
        let output = SurfaceTextureRenderTarget::new(&self.surface)?;
        let mut cmd_encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });
        {
            let mut render_pass = cmd_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: self.hdr.view(),
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLUE),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            self.cubemap
                .render(&mut render_pass, self.camera.bind_group());
            render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));

            render_pass.set_pipeline(&self.light_render_pipeline.pipeline);
            render_pass.draw_light_model(
                &self.model,
                self.camera.bind_group(),
                &self.light_bind_group,
            );

            render_pass.set_pipeline(&self.render_pipeline.pipeline);
            render_pass.draw_model_instanced(
                &self.model,
                self.camera.bind_group(),
                &self.light_bind_group,
                self.cubemap.bind_group(),
                0..self.instances.len() as u32,
            );
        }
        let size = self.surface.extent();
        let screen_descriptor = ScreenDescriptor {
            size_in_pixels: [size.width, size.height],
            pixels_per_point: window.scale_factor() as f32,
        };
        let mut picked_path: Option<PathBuf> = None;
        let mut source = std::mem::take(&mut self.shader_source);
        let mut shader_changed = false;
        let draw_params = EguiDrawParams {
            device: &self.device,
            queue: &self.queue,
            encoder: &mut cmd_encoder,
            window,
            view: self.hdr.view(),
            screen_descriptor,
            run_ui: |ui| {
                egui::Window::new("Shader Editor")
                    .resizable(true)
                    .vscroll(true)
                    .default_open(true)
                    .show(ui, |ui| {
                        if ui.button("Load a new shader").clicked() {
                            if let Some(path) = rfd::FileDialog::new().pick_file() {
                                picked_path = Some(path);
                            }
                        }

                        if let Some(err) = &self.shader_compile_error {
                            ui.label(RichText::new(err).color(Color32::RED));
                        }

                        ui.heading("Shader Source");
                        let mut theme = CodeTheme::from_memory(ui.ctx());
                        ui.collapsing("Theme Settings", |ui| {
                            ui.group(|ui| {
                                theme.ui(ui);
                                theme.clone().store_in_memory(ui.ctx());
                            })
                        });
                        let mut layouter = |ui: &egui::Ui, string: &str, wrap_width: f32| {
                            let mut layout_job = highlight(ui.ctx(), &theme, string, "wgsl");
                            layout_job.wrap.max_width = wrap_width;
                            ui.fonts(|f| f.layout_job(layout_job))
                        };
                        let response = ui.add(
                            egui::TextEdit::multiline(&mut source)
                                .font(egui::TextStyle::Monospace)
                                .code_editor()
                                .lock_focus(true)
                                .desired_width(f32::INFINITY)
                                .layouter(&mut layouter),
                        );
                        response.context_menu(|ui| {
                            if ui.button("Recompile").clicked() {
                                shader_changed = true;
                            }
                            if ui.button("Save").clicked() {
                                if let Some(path) = rfd::FileDialog::new().save_file() {
                                    match std::fs::write(path, &source) {
                                        Ok(_) => (),
                                        Err(e) => error!("Could not save file! {}", e),
                                    }
                                }
                            }
                        });
                        if !shader_changed {
                            shader_changed = response.lost_focus();
                        }
                        ui.horizontal(|ui| {
                            if ui.button("Recompile shader").clicked() {
                                shader_changed = true;
                            }
                            if ui.button("Save shader").clicked() {
                                if let Some(path) = rfd::FileDialog::new().save_file() {
                                    match std::fs::write(path, &source) {
                                        Ok(_) => (),
                                        Err(e) => error!("Could not save file! {}", e),
                                    }
                                }
                            }
                        });
                    });
            },
        };
        self.ui.draw(draw_params);
        self.hdr.process(&mut cmd_encoder, output.view());

        self.queue.submit(std::iter::once(cmd_encoder.finish()));
        output.present();

        // TODO this feels like it should be somewhere else
        self.shader_source = source;
        if let Some(path_to_load) = &picked_path {
            if let Err(e) = self.load_shader(path_to_load) {
                error!("Unable to load shader {}: {}", path_to_load.display(), e);
            }
        }
        if shader_changed {
            info!("Shader changed!");
            self.compile_shader();
        }

        Ok(())
    }
}

pub async fn run() -> RendererResult<()> {
    let event_loop = EventLoop::new().unwrap();
    let window = Arc::new(WindowBuilder::new().build(&event_loop).unwrap());

    let window_id = window.id();
    let mut state = State::new(window.clone(), std::env::args().nth(1)).await?;
    let mut last_render_time = Instant::now();

    Ok(event_loop.run(move |event, control_flow| match event {
        Event::WindowEvent {
            ref event,
            window_id: win_id,
        } if win_id == window_id => {
            if !state.input(&window, event) {
                match event {
                    WindowEvent::CloseRequested
                    | WindowEvent::KeyboardInput {
                        event:
                            KeyEvent {
                                state: ElementState::Pressed,
                                physical_key: PhysicalKey::Code(KeyCode::Escape),
                                ..
                            },
                        ..
                    } => control_flow.exit(),
                    WindowEvent::Resized(physical_size) => state.resize(*physical_size),
                    WindowEvent::RedrawRequested => {
                        let now = Instant::now();
                        let dt = now - last_render_time;
                        state.update(dt);
                        last_render_time = now;
                        match state.render(&window) {
                            Ok(()) => (),
                            Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                            Err(wgpu::SurfaceError::OutOfMemory) => {
                                error!("WGPU Error: Out of Memory!");
                                control_flow.exit();
                            }
                            Err(e) => error!("{:?}", e),
                        }
                    }
                    _ => (),
                }
            }
        }
        Event::DeviceEvent {
            event: DeviceEvent::MouseMotion { delta },
            ..
        } => {
            state.process_mouse_motion(delta.0, delta.1);
        }
        Event::AboutToWait => window.request_redraw(),
        _ => (),
    })?)
}
