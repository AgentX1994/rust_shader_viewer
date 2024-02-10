use egui::{text::LayoutJob, Context};
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

// Everything below here is taken from the egui_extras crate's syntax_highlighting module:
// https://github.com/emilk/egui/blob/master/crates/egui_extras/src/syntax_highlighting.rs
// and lightly edited to load the WGSL sublime syntax file from
// https://github.com/relrelb/sublime-wgsl

/// Add syntax highlighting to a code string.
///
/// The results are memoized, so you can call this every frame without performance penalty.
pub fn highlight(ctx: &egui::Context, theme: &CodeTheme, code: &str, language: &str) -> LayoutJob {
    impl egui::util::cache::ComputerMut<(&CodeTheme, &str, &str), LayoutJob> for Highlighter {
        fn compute(&mut self, (theme, code, lang): (&CodeTheme, &str, &str)) -> LayoutJob {
            self.highlight(theme, code, lang)
        }
    }

    type HighlightCache = egui::util::cache::FrameCache<LayoutJob, Highlighter>;

    ctx.memory_mut(|mem| {
        mem.caches
            .cache::<HighlightCache>()
            .get((theme, code, language))
    })
}

#[derive(Clone, Copy, Hash, PartialEq)]
enum SyntectTheme {
    Base16EightiesDark,
    Base16MochaDark,
    Base16OceanDark,
    Base16OceanLight,
    InspiredGitHub,
    SolarizedDark,
    SolarizedLight,
}

impl SyntectTheme {
    fn all() -> impl ExactSizeIterator<Item = Self> {
        [
            Self::Base16EightiesDark,
            Self::Base16MochaDark,
            Self::Base16OceanDark,
            Self::Base16OceanLight,
            Self::InspiredGitHub,
            Self::SolarizedDark,
            Self::SolarizedLight,
        ]
        .iter()
        .copied()
    }

    fn name(&self) -> &'static str {
        match self {
            Self::Base16EightiesDark => "Base16 Eighties (dark)",
            Self::Base16MochaDark => "Base16 Mocha (dark)",
            Self::Base16OceanDark => "Base16 Ocean (dark)",
            Self::Base16OceanLight => "Base16 Ocean (light)",
            Self::InspiredGitHub => "InspiredGitHub (light)",
            Self::SolarizedDark => "Solarized (dark)",
            Self::SolarizedLight => "Solarized (light)",
        }
    }

    fn syntect_key_name(&self) -> &'static str {
        match self {
            Self::Base16EightiesDark => "base16-eighties.dark",
            Self::Base16MochaDark => "base16-mocha.dark",
            Self::Base16OceanDark => "base16-ocean.dark",
            Self::Base16OceanLight => "base16-ocean.light",
            Self::InspiredGitHub => "InspiredGitHub",
            Self::SolarizedDark => "Solarized (dark)",
            Self::SolarizedLight => "Solarized (light)",
        }
    }

    pub fn is_dark(&self) -> bool {
        match self {
            Self::Base16EightiesDark
            | Self::Base16MochaDark
            | Self::Base16OceanDark
            | Self::SolarizedDark => true,

            Self::Base16OceanLight | Self::InspiredGitHub | Self::SolarizedLight => false,
        }
    }
}

/// A selected color theme.
#[derive(Clone, Hash, PartialEq)]
pub struct CodeTheme {
    dark_mode: bool,
    syntect_theme: SyntectTheme,
}

impl Default for CodeTheme {
    fn default() -> Self {
        Self::dark()
    }
}

impl CodeTheme {
    /// Load code theme from egui memory.
    ///
    /// There is one dark and one light theme stored at any one time.
    pub fn from_memory(ctx: &egui::Context) -> Self {
        if ctx.style().visuals.dark_mode {
            ctx.data_mut(|d| {
                d.get_persisted(egui::Id::new("dark"))
                    .unwrap_or_else(Self::dark)
            })
        } else {
            ctx.data_mut(|d| {
                d.get_persisted(egui::Id::new("light"))
                    .unwrap_or_else(Self::light)
            })
        }
    }

    /// Store theme to egui memory.
    ///
    /// There is one dark and one light theme stored at any one time.
    pub fn store_in_memory(self, ctx: &egui::Context) {
        if self.dark_mode {
            ctx.data_mut(|d| d.insert_persisted(egui::Id::new("dark"), self));
        } else {
            ctx.data_mut(|d| d.insert_persisted(egui::Id::new("light"), self));
        }
    }
}

impl CodeTheme {
    pub fn dark() -> Self {
        Self {
            dark_mode: true,
            syntect_theme: SyntectTheme::Base16MochaDark,
        }
    }

    pub fn light() -> Self {
        Self {
            dark_mode: false,
            syntect_theme: SyntectTheme::SolarizedLight,
        }
    }

    /// Show UI for changing the color theme.
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        egui::widgets::global_dark_light_mode_buttons(ui);

        for theme in SyntectTheme::all() {
            if theme.is_dark() == self.dark_mode {
                ui.radio_value(&mut self.syntect_theme, theme, theme.name());
            }
        }
    }
}

struct Highlighter {
    ps: syntect::parsing::SyntaxSet,
    ts: syntect::highlighting::ThemeSet,
}

impl Default for Highlighter {
    fn default() -> Self {
        Self {
            ps: syntect::parsing::SyntaxSet::load_from_folder(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/res/WGSL.sublime-syntax"
            ))
            .expect("Could not load syntax"),
            ts: syntect::highlighting::ThemeSet::load_defaults(),
        }
    }
}

impl Highlighter {
    #[allow(clippy::unused_self, clippy::unnecessary_wraps)]
    fn highlight(&self, theme: &CodeTheme, code: &str, lang: &str) -> LayoutJob {
        self.highlight_impl(theme, code, lang).unwrap_or_else(|| {
            // Fallback:
            LayoutJob::simple(
                code.into(),
                egui::FontId::monospace(12.0),
                if theme.dark_mode {
                    egui::Color32::LIGHT_GRAY
                } else {
                    egui::Color32::DARK_GRAY
                },
                f32::INFINITY,
            )
        })
    }

    fn highlight_impl(&self, theme: &CodeTheme, text: &str, language: &str) -> Option<LayoutJob> {
        use syntect::easy::HighlightLines;
        use syntect::highlighting::FontStyle;
        use syntect::util::LinesWithEndings;

        let syntax = self
            .ps
            .find_syntax_by_name(language)
            .or_else(|| self.ps.find_syntax_by_extension(language))?;

        let theme = theme.syntect_theme.syntect_key_name();
        let mut h = HighlightLines::new(syntax, &self.ts.themes[theme]);

        use egui::text::{LayoutSection, TextFormat};

        let mut job = LayoutJob {
            text: text.into(),
            ..Default::default()
        };

        for line in LinesWithEndings::from(text) {
            for (style, range) in h.highlight_line(line, &self.ps).ok()? {
                let fg = style.foreground;
                let text_color = egui::Color32::from_rgb(fg.r, fg.g, fg.b);
                let italics = style.font_style.contains(FontStyle::ITALIC);
                let underline = style.font_style.contains(FontStyle::ITALIC);
                let underline = if underline {
                    egui::Stroke::new(1.0, text_color)
                } else {
                    egui::Stroke::NONE
                };
                job.sections.push(LayoutSection {
                    leading_space: 0.0,
                    byte_range: as_byte_range(text, range),
                    format: TextFormat {
                        font_id: egui::FontId::monospace(12.0),
                        color: text_color,
                        italics,
                        underline,
                        ..Default::default()
                    },
                });
            }
        }

        Some(job)
    }
}

fn as_byte_range(whole: &str, range: &str) -> std::ops::Range<usize> {
    let whole_start = whole.as_ptr() as usize;
    let range_start = range.as_ptr() as usize;
    assert!(whole_start <= range_start);
    assert!(range_start + range.len() <= whole_start + whole.len());
    let offset = range_start - whole_start;
    offset..(offset + range.len())
}
