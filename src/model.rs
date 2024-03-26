use std::ops::Range;

use bytemuck::Zeroable;
use cgmath::{Matrix, SquareMatrix};
use log::error;
use wgpu::{util::DeviceExt, vertex_attr_array, VertexAttribute};

use crate::texture;

pub trait Vertex {
    fn layout() -> wgpu::VertexBufferLayout<'static>;
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ModelVertex {
    pub position: [f32; 3],
    pub tex_coords: [f32; 2],
    pub normal: [f32; 3],
    pub tangent: [f32; 3],
    pub bitangent: [f32; 3],
}

impl Vertex for ModelVertex {
    fn layout() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;
        const ATTRIBUTES: [VertexAttribute; 5] = vertex_attr_array![
            // model
            0 => Float32x3,
            1 => Float32x2,
            2 => Float32x3,
            3 => Float32x3,
            4 => Float32x3
        ];
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<ModelVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &ATTRIBUTES,
        }
    }
}

pub struct Mesh {
    pub name: String,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub number_of_elements: u32,
    pub material: usize,
}

pub struct Material {
    pub name: String,
    pub diffuse_texture: texture::Texture,
    pub normal_texture: texture::Texture,
    pub bind_group: wgpu::BindGroup,
}

impl Material {
    pub fn new(
        device: &wgpu::Device,
        name: &str,
        diffuse_texture: texture::Texture,
        normal_texture: texture::Texture,
        layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&normal_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&normal_texture.sampler),
                },
            ],
        });

        Self {
            name: String::from(name),
            diffuse_texture,
            normal_texture,
            bind_group,
        }
    }
}

pub struct InstanceId(usize);

pub struct Model {
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
    pub instances: Vec<InstanceRaw>,
    pub instance_buffer: wgpu::Buffer,
}

impl Model {
    pub fn new_instance(&mut self) -> InstanceId {
        let id = InstanceId(self.instances.len());
        self.instances.push(InstanceRaw::zeroed());
        id
    }

    pub fn update_instance(&mut self, id: &InstanceId, transform: cgmath::Matrix4<f32>) {
        match self.instances.get_mut(id.0) {
            Some(instance) => {
                instance.model = transform.into();
                let inv_trans = match transform.invert() {
                    Some(t) => t.transpose(),
                    None => {
                        error!("Singular matrix!");
                        cgmath::Matrix4::identity()
                    }
                };
                instance.normal = cgmath::Matrix3 {
                    x: cgmath::Vector3 {
                        x: inv_trans.x.x,
                        y: inv_trans.x.y,
                        z: inv_trans.x.z,
                    },
                    y: cgmath::Vector3 {
                        x: inv_trans.y.x,
                        y: inv_trans.y.y,
                        z: inv_trans.y.z,
                    },
                    z: cgmath::Vector3 {
                        x: inv_trans.z.x,
                        y: inv_trans.z.y,
                        z: inv_trans.z.z,
                    },
                }
                .into()
            }
            None => todo!(),
        }
    }

    pub fn update_instance_buffer(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        // Resize buffer if needed
        if self.instances.len() * std::mem::size_of::<InstanceRaw>()
            != self.instance_buffer.size() as usize
        {
            self.instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Instance Buffer"),
                contents: bytemuck::cast_slice(&self.instances),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });
        } else {
            queue.write_buffer(
                &self.instance_buffer,
                0,
                bytemuck::cast_slice(&self.instances),
            );
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InstanceRaw {
    model: [[f32; 4]; 4],
    normal: [[f32; 3]; 3],
}

impl InstanceRaw {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;
        const ATTRIBUTES: [VertexAttribute; 7] = vertex_attr_array![
            // model
            5 => Float32x4,
            6 => Float32x4,
            7 => Float32x4,
            8 => Float32x4,
            9 => Float32x4,
            10 => Float32x4,
            11 => Float32x4,
        ];
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<InstanceRaw>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &ATTRIBUTES,
        }
    }
}

pub trait ModelRenderer<'a> {
    fn draw_mesh(
        &mut self,
        mesh: &'a Mesh,
        material: &'a Material,
        camera_bind_group: &'a wgpu::BindGroup,
        light_bind_group: &'a wgpu::BindGroup,
        environment_bind_group: &'a wgpu::BindGroup,
    ) {
        self.draw_mesh_instanced(
            mesh,
            material,
            camera_bind_group,
            light_bind_group,
            environment_bind_group,
            0..1,
        );
    }

    fn draw_mesh_instanced(
        &mut self,
        mesh: &'a Mesh,
        material: &'a Material,
        camera_bind_group: &'a wgpu::BindGroup,
        light_bind_group: &'a wgpu::BindGroup,
        environment_bind_group: &'a wgpu::BindGroup,
        instances: Range<u32>,
    );

    fn draw_model(
        &mut self,
        model: &'a Model,
        camera_bind_group: &'a wgpu::BindGroup,
        light_bind_group: &'a wgpu::BindGroup,
        environment_bind_group: &'a wgpu::BindGroup,
    ) {
        self.draw_model_instanced(
            model,
            camera_bind_group,
            light_bind_group,
            environment_bind_group,
            0..1,
        );
    }

    fn draw_model_instanced(
        &mut self,
        model: &'a Model,
        camera_bind_group: &'a wgpu::BindGroup,
        light_bind_group: &'a wgpu::BindGroup,
        environment_bind_group: &'a wgpu::BindGroup,
        instances: Range<u32>,
    );
}

impl<'a, 'b> ModelRenderer<'b> for wgpu::RenderPass<'a>
where
    'b: 'a,
{
    fn draw_mesh_instanced(
        &mut self,
        mesh: &'b Mesh,
        material: &'b Material,
        camera_bind_group: &'b wgpu::BindGroup,
        light_bind_group: &'b wgpu::BindGroup,
        environment_bind_group: &'b wgpu::BindGroup,
        instances: Range<u32>,
    ) {
        self.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        self.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        self.set_bind_group(0, &material.bind_group, &[]);
        self.set_bind_group(1, camera_bind_group, &[]);
        self.set_bind_group(2, light_bind_group, &[]);
        self.set_bind_group(3, environment_bind_group, &[]);
        self.draw_indexed(0..mesh.number_of_elements, 0, instances);
    }

    fn draw_model_instanced(
        &mut self,
        model: &'b Model,
        camera_bind_group: &'b wgpu::BindGroup,
        light_bind_group: &'b wgpu::BindGroup,
        environment_bind_group: &'b wgpu::BindGroup,
        instances: Range<u32>,
    ) {
        self.set_vertex_buffer(1, model.instance_buffer.slice(..));
        for mesh in &model.meshes {
            let material = &model.materials[mesh.material];
            self.draw_mesh_instanced(
                mesh,
                material,
                camera_bind_group,
                light_bind_group,
                environment_bind_group,
                instances.clone(),
            );
        }
    }
}

pub trait LightRenderer<'a> {
    fn draw_light_mesh(
        &mut self,
        mesh: &'a Mesh,
        camera_bind_group: &'a wgpu::BindGroup,
        light_bind_group: &'a wgpu::BindGroup,
    ) {
        self.draw_light_mesh_instanced(mesh, camera_bind_group, light_bind_group, 0..1);
    }

    fn draw_light_mesh_instanced(
        &mut self,
        mesh: &'a Mesh,
        camera_bind_group: &'a wgpu::BindGroup,
        light_bind_group: &'a wgpu::BindGroup,
        instances: Range<u32>,
    );

    fn draw_light_model(
        &mut self,
        model: &'a Model,
        camera_bind_group: &'a wgpu::BindGroup,
        light_bind_group: &'a wgpu::BindGroup,
    ) {
        self.draw_light_model_instanced(model, camera_bind_group, light_bind_group, 0..1);
    }

    fn draw_light_model_instanced(
        &mut self,
        model: &'a Model,
        camera_bind_group: &'a wgpu::BindGroup,
        light_bind_group: &'a wgpu::BindGroup,
        instances: Range<u32>,
    );
}

impl<'a, 'b> LightRenderer<'b> for wgpu::RenderPass<'a>
where
    'b: 'a,
{
    fn draw_light_mesh_instanced(
        &mut self,
        mesh: &'b Mesh,
        camera_bind_group: &'b wgpu::BindGroup,
        light_bind_group: &'b wgpu::BindGroup,
        instances: Range<u32>,
    ) {
        self.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        self.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        self.set_bind_group(0, camera_bind_group, &[]);
        self.set_bind_group(1, light_bind_group, &[]);
        self.draw_indexed(0..mesh.number_of_elements, 0, instances);
    }

    fn draw_light_model_instanced(
        &mut self,
        model: &'b Model,
        camera_bind_group: &'b wgpu::BindGroup,
        light_bind_group: &'b wgpu::BindGroup,
        instances: Range<u32>,
    ) {
        self.set_vertex_buffer(1, model.instance_buffer.slice(..));
        for mesh in &model.meshes {
            self.draw_light_mesh_instanced(
                mesh,
                camera_bind_group,
                light_bind_group,
                instances.clone(),
            );
        }
    }
}
