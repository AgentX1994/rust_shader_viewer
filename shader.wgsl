// Vertex Shader

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) tangent: vec3<f32>,
    @location(4) bitangent: vec3<f32>,
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
    @location(1) tangent_position: vec3<f32>,
    @location(2) tangent_light_position: vec3<f32>,
    @location(3) tangent_view_position: vec3<f32>,
};

struct CameraUniform {
    view_pos: vec4<f32>,
    view_proj: mat4x4<f32>,
};

@group(1) @binding(0)
var<uniform> camera: CameraUniform;

struct Light {
    position: vec3<f32>,
    color: vec3<f32>
};

@group(2) @binding(0)
var<uniform> light: Light;

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

    let world_normal = normalize(normal_mat * input.normal);
    let world_tangent = normalize(normal_mat * input.tangent);
    let world_bitangent = normalize(normal_mat * input.bitangent);
    let tangent_matrix = transpose(mat3x3<f32>(
        world_tangent,
        world_bitangent,
        world_normal
    ));
    let world_position = model_mat * vec4<f32>(input.position, 1.0);

    out.clip_position = camera.view_proj * world_position;
    out.tex_coords = input.tex_coords;
    out.tangent_position = tangent_matrix * world_position.xyz;
    out.tangent_view_position = tangent_matrix * camera.view_pos.xyz;
    out.tangent_light_position = tangent_matrix * light.position;
    return out;
}

@group(0) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(0) @binding(1)
var s_diffuse: sampler;

@group(0) @binding(2)
var t_normal: texture_2d<f32>;
@group(0) @binding(3)
var s_normal: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let obj_color = textureSample(t_diffuse, s_diffuse, in.tex_coords);
    let obj_normal: vec4<f32> = textureSample(t_normal, s_normal, in.tex_coords);

    let ambient_strength = 0.1;
    let ambient_color = light.color * ambient_strength;

    let tangent_normal = obj_normal.xyz * 2.0 - 1.0;
    let light_dir = normalize(in.tangent_light_position - in.tangent_position);
    let view_dir = normalize(in.tangent_view_position - in.tangent_position);
    let half_dir = normalize(view_dir + light_dir);

    let diffuse_strength = max(dot(tangent_normal, light_dir), 0.0);
    let diffuse_color = light.color * diffuse_strength;

    //let reflect_dir = reflect(-light_dir, in.world_normal);

    let specular_strength = pow(max(dot(tangent_normal, half_dir), 0.0), 32.0);
    let specular_color = specular_strength * light.color;

    let final_color = (ambient_color + diffuse_color + specular_color) * obj_color.xyz;
    return vec4<f32>(final_color, 1.0);
}
