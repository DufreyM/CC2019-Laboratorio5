use std::io::Cursor;
use std::time::{Instant, Duration};
use std::fs::File;
use std::path::Path;

use glium::{glutin, Surface, uniform};
use cgmath::{Matrix4, Rad, perspective, Point3, Vector3};
use tobj;

#[derive(Copy, Clone)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
}
glium::implement_vertex!(Vertex, position, normal);

fn load_obj_vertices(path: &str) -> (Vec<Vertex>, Vec<u32>) {
    let (models, _materials) = tobj::load_obj(path, &tobj::LoadOptions {
        single_index: true,
        triangulate: true,
        ..Default::default()
    }).expect("Failed to load OBJ");
    let mut vertices: Vec<Vertex> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    for m in models.iter() {
        let mesh = &m.mesh;
        let positions = &mesh.positions;
        let normals = if mesh.normals.len() > 0 {
            Some(&mesh.normals)
        } else {
            None
        };

        for i in 0..(positions.len() / 3) {
            let p = [positions[3*i], positions[3*i+1], positions[3*i+2]];
            let n = if let Some(ns) = normals {
                [ns[3*i], ns[3*i+1], ns[3*i+2]]
            } else {
                [0.0, 1.0, 0.0]
            };
            vertices.push(Vertex { position: p, normal: n });
        }

        for &idx in mesh.indices.iter() {
            indices.push(idx as u32);
        }
    }
    (vertices, indices)
}

fn main() {
    // window + context
    let event_loop = glutin::event_loop::EventLoop::new();
    let wb = glutin::window::WindowBuilder::new()
        .with_title("Star shader - time + noise")
        .with_inner_size(glutin::dpi::LogicalSize::new(1024.0, 768.0));
    let cb = glutin::ContextBuilder::new().with_depth_buffer(24);
    let display = glium::Display::new(wb, cb, &event_loop).unwrap();

    // load sphere.obj
    let asset_path = "assets/sphere.obj";
    if !Path::new(asset_path).exists() {
        panic!("Put your sphere.obj in assets/sphere.obj");
    }
    let (vertices, indices) = load_obj_vertices(asset_path);
    let vertex_buffer = glium::VertexBuffer::new(&display, &vertices).unwrap();
    let index_buffer = glium::IndexBuffer::new(
        &display,
        glium::index::PrimitiveType::TrianglesList,
        &indices
    ).unwrap();

    // shaders (vertex + fragment). Usamos la versión GLSL compatible con glium (330 core)
    let vertex_shader_src = r#"
        #version 330 core

        in vec3 position;
        in vec3 normal;

        uniform mat4 uModel;
        uniform mat4 uView;
        uniform mat4 uProj;

        uniform float time;
        uniform float noiseScale;
        uniform float noiseAmplitude;
        uniform float vertexTwist;

        out vec3 vWorldPos;
        out vec3 vNormal;
        out float vNoise;

        // --- small perlin-ish noise (same as provided earlier but shortened) ---
        // For brevity we use a compact value-noise (good enough visually).
        vec3 mod289(vec3 x){ return x - floor(x * (1.0/289.0)) * 289.0; }
        vec4 mod289(vec4 x){ return x - floor(x * (1.0/289.0)) * 289.0; }
        vec4 permute(vec4 x){ return mod289(((x*34.0)+1.0)*x); }
        vec4 taylorInvSqrt(vec4 r){ return 1.79284291400159 - 0.85373472095314 * r; }
        float cnoise(vec3 P) {
            vec3 Pi0 = floor(P);
            vec3 Pi1 = Pi0 + vec3(1.0);
            Pi0 = mod289(Pi0);
            Pi1 = mod289(Pi1);
            vec3 Pf0 = fract(P);
            vec3 Pf1 = Pf0 - vec3(1.0);

            vec4 ix = vec4(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
            vec4 iy = vec4(Pi0.y, Pi0.y, Pi1.y, Pi1.y);
            vec4 iz0 = vec4(Pi0.z);
            vec4 iz1 = vec4(Pi1.z);

            vec4 ixy = permute(permute(ix) + iy);
            vec4 ixy0 = permute(ixy + iz0);
            vec4 ixy1 = permute(ixy + iz1);

            vec4 gx0 = ixy0 * (1.0/7.0);
            vec4 gy0 = fract(floor(gx0) * (1.0/7.0)) - 0.5;
            gx0 = fract(gx0);
            vec4 gz0 = vec4(0.5) - abs(gx0) - abs(gy0);
            vec4 sz0 = step(gz0, vec4(0.0));
            gx0 -= sz0 * (step(0.0, gx0) - 0.5);
            gy0 -= sz0 * (step(0.0, gy0) - 0.5);

            vec4 gx1 = ixy1 * (1.0/7.0);
            vec4 gy1 = fract(floor(gx1) * (1.0/7.0)) - 0.5;
            gx1 = fract(gx1);
            vec4 gz1 = vec4(0.5) - abs(gx1) - abs(gy1);
            vec4 sz1 = step(gz1, vec4(0.0));
            gx1 -= sz1 * (step(0.0, gx1) - 0.5);
            gy1 -= sz1 * (step(0.0, gy1) - 0.5);

            vec3 g000 = vec3(gx0.x,gy0.x,gz0.x);
            vec3 g100 = vec3(gx0.y,gy0.y,gz0.y);
            vec3 g010 = vec3(gx0.z,gy0.z,gz0.z);
            vec3 g110 = vec3(gx0.w,gy0.w,gz0.w);
            vec3 g001 = vec3(gx1.x,gy1.x,gz1.x);
            vec3 g101 = vec3(gx1.y,gy1.y,gz1.y);
            vec3 g011 = vec3(gx1.z,gy1.z,gz1.z);
            vec3 g111 = vec3(gx1.w,gy1.w,gz1.w);

            vec4 norm0 = taylorInvSqrt(vec4(dot(g000,g000), dot(g010,g010), dot(g100,g100), dot(g110,g110)));
            g000 *= norm0.x; g010 *= norm0.y; g100 *= norm0.z; g110 *= norm0.w;
            vec4 norm1 = taylorInvSqrt(vec4(dot(g001,g001), dot(g011,g011), dot(g101,g101), dot(g111,g111)));
            g001 *= norm1.x; g011 *= norm1.y; g101 *= norm1.z; g111 *= norm1.w;

            float n000 = dot(g000, Pf0);
            float n100 = dot(g100, vec3(Pf1.x, Pf0.yz));
            float n010 = dot(g010, vec3(Pf0.x, Pf1.y, Pf0.z));
            float n110 = dot(g110, vec3(Pf1.x, Pf1.y, Pf0.z));
            float n001 = dot(g001, vec3(Pf0.xy, Pf1.z));
            float n101 = dot(g101, vec3(Pf1.x, Pf0.y, Pf1.z));
            float n011 = dot(g011, vec3(Pf0.x, Pf1.yz));
            float n111 = dot(g111, Pf1);

            vec3 fade_xyz = Pf0 * Pf0 * Pf0 * (Pf0 * (Pf0 * 6.0 - 15.0) + 10.0);
            vec4 n_z = mix(vec4(n000, n100, n010, n110), vec4(n001, n101, n011, n111), fade_xyz.z);
            vec2 n_y = mix(n_z.xy, n_z.zw, fade_xyz.y);
            return mix(n_y.x, n_y.y, fade_xyz.x);
        }

        void main() {
            // position in model space
            vec3 pos = position;
            // sample noise at position scaled by noiseScale and animated by time
            float noise = cnoise(vec3(pos * noiseScale) + vec3(0.0, time * 0.4, 0.0));
            // additional octaves for richer surface
            float n2 = cnoise(vec3(pos * noiseScale * 2.0) + vec3(time * 0.9));
            float n3 = cnoise(vec3(pos * noiseScale * 4.0) - vec3(time * 1.3));
            float combined = noise * 0.6 + n2 * 0.3 + n3 * 0.1;
            // displacement outward along normal (flare / turbulence)
            vec3 displaced = pos + normal * combined * noiseAmplitude;

            // small twist based on vertexTwist and time to simulate pulses
            displaced += normal * sin(time*2.0 + length(pos)) * (vertexTwist * 0.02);

            vNoise = combined;

            vec4 world_pos = uModel * vec4(displaced, 1.0);
            vWorldPos = world_pos.xyz;
            // approximate normal: transform original normal by model (no normal matrix for brevity)
            vNormal = mat3(uModel) * normal;

            gl_Position = uProj * uView * world_pos;
        }
    "#;

    let fragment_shader_src = r#"
        #version 330 core

        in vec3 vWorldPos;
        in vec3 vNormal;
        in float vNoise;

        out vec4 color;

        uniform vec3 viewPos;
        uniform float time;
        uniform float emissionBoost;   // multiplier for emission intensity
        uniform float pulseAmp;        // how strong pulses are
        uniform float tempCold;        // color gradient low
        uniform float tempHot;         // color gradient high

        // helper: map value [0,1] to a color gradient (cool -> hot)
        vec3 gradient_color(float t) {
            // t in [0,1]
            // cool: deep orange -> yellow -> white for hot
            vec3 cold = vec3(0.8, 0.3, 0.05); // deep orange
            vec3 mid  = vec3(1.0, 0.6, 0.1);  // orange/yellow
            vec3 hot  = vec3(1.0, 1.0, 0.9);  // near white
            if (t < 0.5) {
                return mix(cold, mid, t*2.0);
            } else {
                return mix(mid, hot, (t-0.5)*2.0);
            }
        }

        void main() {
            // base intensity from noise (normalized)
            float base = clamp(vNoise * 1.3 + 0.5, 0.0, 1.0);
            // pulsation using time
            float pulse = 0.5 + 0.5 * sin(time * 3.0 + vNoise * 10.0) * pulseAmp;
            float intensity = base * pulse * emissionBoost;

            // color from gradient according to intensity (simulate temperature variation)
            vec3 col = gradient_color(intensity);

            // simple diffuse + emission-like final color
            vec3 N = normalize(vNormal);
            vec3 L = normalize(vec3(0.0, 0.0, 1.0)); // fake light toward camera
            float diff = max(dot(N, L), 0.0);

            // final combines diffuse (gives shading) and emission (makes it glow)
            vec3 final_col = col * (0.3 * diff + 0.7) * (1.0 + intensity*0.8);

            // tone mapping / clamp to avoid overflow
            final_col = final_col / (final_col + vec3(1.0));
            color = vec4(final_col, 1.0);
        }
    "#;

    // compile program
    let program = glium::Program::from_source(&display, vertex_shader_src, fragment_shader_src, None).unwrap();

    // uniforms (initial)
    let start = Instant::now();

    // matrices
    let mut t: f32 = 0.0;
    let mut closed = false;

    // camera & matrices
    let projection: [[f32; 4]; 4] = {
        let proj: Matrix4<f32> = perspective(Rad(std::f32::consts::FRAC_PI_3), 1024.0/768.0, 0.1, 100.0);
        proj.into()
    };
    let view: [[f32; 4]; 4] = {
        let eye = Point3::new(0.0, 0.0, 3.0);
        let target = Point3::new(0.0, 0.0, 0.0);
        let up = Vector3::new(0.0, 1.0, 0.0);
        let view_mat = Matrix4::look_at_rh(eye, target, up);
        view_mat.into()
    };
    let model: [[f32; 4]; 4] = Matrix4::from_scale(1.0).into();

    // event loop
    event_loop.run(move |ev, _, control_flow| {
        // time
        let elapsed = start.elapsed();
        let time_secs = elapsed.as_secs_f32();

        // handle events
        match ev {
            glutin::event::Event::WindowEvent { event, .. } => match event {
                glutin::event::WindowEvent::CloseRequested => {
                    *control_flow = glutin::event_loop::ControlFlow::Exit;
                    return;
                },
                glutin::event::WindowEvent::KeyboardInput { input, .. } => {
                    if let Some(glutin::event::VirtualKeyCode::Escape) = input.virtual_keycode {
                        *control_flow = glutin::event_loop::ControlFlow::Exit;
                        return;
                    }
                }
                _ => {}
            },
            glutin::event::Event::MainEventsCleared => {
                // draw frame
                let mut target = display.draw();
                target.clear_color_and_depth((0.0, 0.0, 0.0, 1.0), 1.0);

                // build uniforms
                let uniforms = uniform! {
                    uModel: model,
                    uView: view,
                    uProj: projection,
                    time: time_secs,
                    noiseScale: 2.5f32,
                    noiseAmplitude: 0.22f32,
                    vertexTwist: 1.0f32,
                    viewPos: [0.0f32, 0.0f32, 3.0f32],
                    emissionBoost: 1.0f32,
                    pulseAmp: 0.9f32,
                    tempCold: 2000.0f32,
                    tempHot: 6000.0f32,
                };

                let params = glium::DrawParameters {
                    depth: glium::Depth {
                        test: glium::draw_parameters::DepthTest::IfLess,
                        write: true,
                        ..Default::default()
                    },
                    backface_culling: glium::draw_parameters::BackfaceCullingMode::CullCounterClockwise,
                    ..Default::default()
                };

                target.draw(&vertex_buffer, &index_buffer, &program, &uniforms, &params).unwrap();
                target.finish().unwrap();

                // request next frame
                *control_flow = glutin::event_loop::ControlFlow::Poll;
            },
            _ => {}
        }
    });
}
