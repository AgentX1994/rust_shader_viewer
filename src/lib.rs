use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use egui_wgpu::ScreenDescriptor;
use winit::{
    event::*,
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowBuilder},
};

use cgmath::prelude::*;
use log::{debug, error, info};
use wgpu::util::DeviceExt;

mod camera;
mod cubemap;
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
use ui::{EguiDrawParams, EguiRenderer};

struct State {
    surface: Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    size: winit::dpi::PhysicalSize<u32>,
    camera: PerspectiveCamera,
    camera_controller: CameraController,
    mouse_pressed: bool,
    depth_texture: texture::Texture,
    render_pipeline: RenderPipeline,
    alt_render_pipeline: Option<RenderPipeline>,
    current_pipeline: u32,
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
    async fn new(window: Arc<Window>, extra_shader: Option<String>) -> Self {
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
                    required_limits: wgpu::Limits::downlevel_defaults(),
                },
                None,
            )
            .await
            .unwrap();

        let surface = Surface::new((size.width, size.height), surface, &adapter, &device);
        debug!(
            "Surface created, surface: {:?}, capabilities: {:?}",
            surface.surface(),
            surface.capabilities()
        );
        let surface_size = surface.extent();

        let depth_texture =
            texture::Texture::create_depth_texture(&device, &surface_size, "depth_texture");

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Bind Group Layout"),
                entries: &[
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
                ],
            });

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

        let light_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Light Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

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
            camera.layout(),
            hdr.format(),
            "pure-sky.hdr",
            Some("Sky Cubemap"),
        )
        .await;

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

        let render_pipeline = {
            let shader = Shader::new_wgsl(
                &device,
                "normal",
                "vs_main",
                "fs_main",
                include_str!("../shader.wgsl"),
            );
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
                "vs_main",
                "fs_main",
                include_str!("../light_shader.wgsl"),
            );
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
        let alt_render_pipeline = extra_shader.map(|shader_file_path| {
            let source =
                std::fs::read_to_string(shader_file_path).expect("Could not read extra shader!");
            // TODO: get entry points from shader source?
            let shader = Shader::new_wgsl(&device, "alt shader", "vs_main", "fs_main", &source);
            RenderPipeline::new(
                &device,
                &render_pipeline_layout,
                hdr.format(),
                Some(texture::Texture::DEPTH_FORMAT),
                &[model::ModelVertex::layout(), Instance::desc()],
                wgpu::PrimitiveTopology::TriangleList,
                &shader,
            )
        });
        let ui = EguiRenderer::new(&device, hdr.format(), None, 1, &window);
        Self {
            surface,
            device,
            queue,
            size,
            camera,
            camera_controller: CameraController::new(4.0, 1.0),
            mouse_pressed: false,
            depth_texture,
            render_pipeline,
            alt_render_pipeline,
            current_pipeline: 0,
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
        }
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
            } => {
                if !self.camera_controller.process_keyboard(*key, *state) {
                    if *state == ElementState::Pressed && *key == KeyCode::KeyP {
                        // Switch pipelines
                        self.current_pipeline ^= 1;
                        true
                    } else {
                        false
                    }
                } else {
                    true
                }
            }
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

            if self.current_pipeline == 0 {
                render_pass.set_pipeline(&self.render_pipeline.pipeline);
            } else {
                match &self.alt_render_pipeline {
                    Some(pipeline) => render_pass.set_pipeline(&pipeline.pipeline),
                    None => render_pass.set_pipeline(&self.render_pipeline.pipeline),
                }
            }
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
        let draw_params = EguiDrawParams {
            device: &self.device,
            queue: &self.queue,
            encoder: &mut cmd_encoder,
            window,
            view: self.hdr.view(),
            screen_descriptor,
            run_ui: |ui| {
                egui::Window::new("Test")
                    .resizable(true)
                    .vscroll(true)
                    .default_open(true)
                    .show(ui, |ui| {
                        ui.label("Window!");
                        ui.label("Window!");
                        ui.label("Window!");
                        ui.label("Window!");
                    });
            },
        };
        self.ui.draw(draw_params);
        self.hdr.process(&mut cmd_encoder, output.view());

        self.queue.submit(std::iter::once(cmd_encoder.finish()));
        output.present();

        Ok(())
    }
}

pub async fn run() {
    let extra_shader = std::env::args().nth(1);
    let event_loop = EventLoop::new().unwrap();
    let window = Arc::new(WindowBuilder::new().build(&event_loop).unwrap());

    let window_id = window.id();
    let mut state = State::new(window.clone(), extra_shader).await;
    let mut last_render_time = Instant::now();

    let res = event_loop.run(move |event, control_flow| match event {
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
    });
    res.expect("Error running event loop!");
}
