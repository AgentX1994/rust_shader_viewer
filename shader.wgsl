// Vertex Shader

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
    @location(2) normal: vec3<f32>,
};

struct InstanceInput {
    @location(5) model_mat_0: vec4<f32>,
    @location(6) model_mat_1: vec4<f32>,
    @location(7) model_mat_2: vec4<f32>,
    @location(8) model_mat_3: vec4<f32>,

    @location(9)  normal_mat_0: vec3<f32>,
    @location(10) normal_mat_1: vec3<f32>,
    @location(11) normal_mat_2: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) world_position: vec3<f32>,
};

struct CameraUniform {
    view_pos: vec4<f32>,
    view_proj: mat4x4<f32>,
};

@group(1) @binding(0)
var<uniform> camera: CameraUniform;

@vertex
fn vs_main(
    input: VertexInput,
    instance: InstanceInput
) -> VertexOutput {
    var out: VertexOutput;
    let model_mat = mat4x4(
        instance.model_mat_0,
        instance.model_mat_1,
        instance.model_mat_2,
        instance.model_mat_3,
    );
    let normal_mat = mat3x3(
        instance.normal_mat_0,
        instance.normal_mat_1,
        instance.normal_mat_2,
    );
    let world_position = model_mat * vec4<f32>(input.position, 1.0);
    out.clip_position = camera.view_proj * world_position;
    out.tex_coords = input.tex_coords;
    out.world_normal = normal_mat * input.normal;
    out.world_position = world_position.xyz;
    return out;
}

@group(0) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(0) @binding(1)
var s_diffuse: sampler;

struct Light {
    position: vec3<f32>,
    color: vec3<f32>
};

@group(2) @binding(0)
var<uniform> light: Light;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let obj_color = textureSample(t_diffuse, s_diffuse, in.tex_coords);

    let ambient_strength = 0.1;
    let ambient_color = light.color * ambient_strength;

    let light_dir = normalize(light.position - in.world_position);

    let diffuse_strength = max(dot(in.world_normal, light_dir), 0.0);
    let diffuse_color = light.color * diffuse_strength;

    let view_dir = normalize(camera.view_pos.xyz - in.world_position);
    //let reflect_dir = reflect(-light_dir, in.world_normal);
    let half_dir = normalize(view_dir + light_dir);

    let specular_strength = pow(max(dot(in.world_normal, half_dir), 0.0), 32.0);
    let specular_color = specular_strength * light.color;

    let final_color = (ambient_color * diffuse_color + specular_color) * obj_color.xyz;
    return vec4<f32>(final_color, 1.0);
}
