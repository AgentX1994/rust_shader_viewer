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
    @location(1) world_position: vec3<f32>,
    @location(2) world_view_position: vec3<f32>,
    @location(3) world_light_position: vec3<f32>,
    @location(4) world_normal: vec3<f32>,
    @location(5) world_tangent: vec3<f32>,
    @location(6) world_bitangent: vec3<f32>,
};

struct CameraUniform {
    view_pos: vec4<f32>,
    view: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    inv_view: mat4x4<f32>,
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

    var out: VertexOutput;
    out.clip_position = camera.view_proj * world_position;
    out.tex_coords = input.tex_coords;
    out.world_position = world_position.xyz;
    out.world_view_position = camera.view_pos.xyz;
    out.world_light_position = light.position;
    out.world_normal = normalize(normal_mat * input.normal);
    out.world_tangent = normalize(normal_mat * input.tangent);
    out.world_bitangent = normalize(normal_mat * input.bitangent);
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

@group(3) @binding(0)
var env_map: texture_cube<f32>;
@group(3) @binding(1)
var env_sampler: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let obj_color = textureSample(t_diffuse, s_diffuse, in.tex_coords);
    let obj_normal: vec4<f32> = textureSample(t_normal, s_normal, in.tex_coords);

    // Fix the tangent and bitangent using the gramm-schmidt process
    let world_tangent = normalize(in.world_tangent - dot(in.world_tangent, in.world_normal) * in.world_normal);
    let world_bitangent = cross(world_tangent, in.world_normal);

    // Convert the normal space to world space
    let TBN = mat3x3(
        world_tangent,
        world_bitangent,
        in.world_normal
    );
    let tangent_normal = obj_normal.xyz * 2.0 - 1.0;
    let world_normal = TBN*tangent_normal;

    let ambient_strength = 0.1;
    let ambient_color = light.color * ambient_strength;

    let light_dir = normalize(in.world_light_position - in.world_position);
    let view_dir = normalize(in.world_view_position - in.world_position);
    let half_dir = normalize(view_dir + light_dir);

    let diffuse_strength = max(dot(world_normal, light_dir), 0.0);
    let diffuse_color = light.color * diffuse_strength;

    let specular_strength = pow(max(dot(world_normal, half_dir), 0.0), 32.0);
    let specular_color = specular_strength * light.color;

    // Calculate reflections
    let world_reflect = reflect(-view_dir, world_normal);
    let reflection = textureSample(env_map, env_sampler, world_reflect).rgb;
    let shininess = 0.1;

    let final_color = (ambient_color + diffuse_color + specular_color) * obj_color.xyz + reflection * shininess;
    return vec4<f32>(final_color, obj_color.a);
}
