#ifndef HDRPIX_H

#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#define HDRPIX_NCC (3) // number of color components

static void chkgl(const char* file, const int line)
{
	GLenum err = glGetError();
	if (err != GL_NO_ERROR) {
		fprintf(stderr, "OPENGL ERROR 0x%.4x in %s:%d\n", err, file, line);
		abort();
	}
}
#define CHKGL chkgl(__FILE__, __LINE__);

enum hdrpix_shader {
	HDRPIX_SHADER_NULL = 0,
	HDRPIX_SHADER_NOISY,
	HDRPIX_SHADER_MAX
};

struct hdrpix_shader_config {
	enum hdrpix_shader shader;
	union {
		struct {
		} noisy;
	};
};

struct hdrpix__tex {
	int width;
	int height;
	GLuint texture;
	int is_initialized;
};

struct hdrpix__fb {
	struct hdrpix__tex tex;
	GLuint framebuffer;
	int is_initialized;
};

struct hdrpix__uniform {
	char* name;
	int type;
	int element_count;
	void* offset;
	GLint location;
};

#define HDRPIX__PRG_MAX_UNIFORMS (8)

struct hdrpix__prg {
	GLuint program;
	int n_uniforms;
	struct hdrpix__uniform uniforms[HDRPIX__PRG_MAX_UNIFORMS];
};

struct hdrpix {
	int config_width;
	int config_height;
	float max_luminance;
	int canvas_width;
	int canvas_height;
	unsigned char* canvas;
	struct hdrpix__tex canvas_tex;

	int display_width;
	int display_height;

	struct hdrpix_shader_config current_shader_config;
	union {
		struct {
			struct hdrpix__fb  fb;
			struct hdrpix__prg prg;
		} noisy;
	} ss;
	struct hdrpix__prg pix_prg;
};

struct hdrpix__pix_uniforms {
	int u_src_texture;
	float u_src_resolution[2];
	float u_d0[2];
	float u_d1[2];
	float u_scale;
};

struct hdrpix__noisy_uniforms {
	int u_src_texture;
	float u_src_scale[2];
	float u_max_luminance;
	float u_seed;
};

static int hdrpix__tex_setup(struct hdrpix__tex* tex, int width, int height)
{
	if (!tex->is_initialized) {
		glGenTextures(1, &tex->texture); CHKGL;
		tex->is_initialized = 1;
	}
	if (tex->width == width && tex->height == height) return 0;
	tex->width = width;
	tex->height = height;
	glBindTexture(GL_TEXTURE_2D, tex->texture); CHKGL;
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); CHKGL;
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); CHKGL;
	glTexImage2D(GL_TEXTURE_2D, /*level=*/0, GL_RGB, tex->width, tex->height, /*border=*/0, GL_RGB, GL_UNSIGNED_BYTE, NULL); CHKGL;
	glBindTexture(GL_TEXTURE_2D, 0); CHKGL;
	return 1;
}

static void hdrpix__tex_upload(struct hdrpix__tex* tex, const void* pixels)
{
	if (tex->width == 0 || tex->height == 0) return;
	glBindTexture(GL_TEXTURE_2D, tex->texture); CHKGL;
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1); CHKGL;
	glTexSubImage2D(GL_TEXTURE_2D, /*level=*/0, /*xOffset=*/0, /*yOffset=*/0, tex->width, tex->height, GL_RGB, GL_UNSIGNED_BYTE, pixels); CHKGL;
	glBindTexture(GL_TEXTURE_2D, 0); CHKGL;
}

static void hdrpix__tex_free(struct hdrpix__tex* tex)
{
	if (!tex->is_initialized) return
	glDeleteTextures(1, &tex->texture); CHKGL;
}

static void hdrpix__fb_bind(struct hdrpix__fb* fb, int width, int height)
{
	if (!fb->is_initialized) {
		fb->is_initialized = 1;
		glGenFramebuffers(1, &fb->framebuffer); CHKGL;
	}

	hdrpix__tex_setup(&fb->tex, width, height);

	glBindFramebuffer(GL_FRAMEBUFFER, fb->framebuffer); CHKGL;
	glViewport(0, 0, width, height);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fb->tex.texture, /*level=*/0); CHKGL;
	assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
}

static void hdrpix__fb_free(struct hdrpix__fb* fb)
{
	if (!fb->is_initialized) return;
	glDeleteFramebuffers(1, &fb->framebuffer); CHKGL;
	hdrpix__tex_free(&fb->tex);
	fb->is_initialized = 0;
}

enum {
	HDRPIX__FLOAT = 1,
	HDRPIX__INT   = 2,
};

#define HDRPIX__ARRAY_LENGTH(x) (sizeof(x) / sizeof(x[0]))
#define HDRPIX__MEMBER_SIZE(t,m) sizeof(((t *)0)->m)
#define HDRPIX__MEMBER_OFFSET(t,m) (void*)((size_t)&(((t *)0)->m))

#define HDRPIX__UNIFORM_FLOATS(t,m) {#m, HDRPIX__FLOAT, HDRPIX__MEMBER_SIZE(t,m)/sizeof(float), HDRPIX__MEMBER_OFFSET(t,m)}
#define HDRPIX__UNIFORM_INTS(t,m)   {#m, HDRPIX__INT,   HDRPIX__MEMBER_SIZE(t,m)/sizeof(int),   HDRPIX__MEMBER_OFFSET(t,m)}

#define HDRPIX__IS_Q0 "(gl_VertexID == 0 || gl_VertexID == 3)"
#define HDRPIX__IS_Q1 "(gl_VertexID == 1)"
#define HDRPIX__IS_Q2 "(gl_VertexID == 2 || gl_VertexID == 4)"
#define HDRPIX__IS_Q3 "(gl_VertexID == 5)"

// https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
// cheap, but oversaturates?
#define HDRPIX__TONEMAP_ACES0                                          \
	"vec3 TONEMAP(vec3 x)\n"                                       \
	"{\n"                                                          \
	"	float a = 2.51;\n"                                     \
	"	float b = 0.03;\n"                                     \
	"	float c = 2.43;\n"                                     \
	"	float d = 0.59;\n"                                     \
	"	float e = 0.14;\n"                                     \
	"	return clamp((x*(a*x+b))/(x*(c*x+d)+e), 0.0, 1.0);\n"  \
	"}\n"

// https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl
#define HDRPIX__TONEMAP_ACES1                                                                                \
	"mat3 ACES1_input_mat = mat3(\n"                                                                     \
	"	vec3(0.59719, 0.35458, 0.04823),\n"                                                          \
	"	vec3(0.07600, 0.90834, 0.01566),\n"                                                          \
	"	vec3(0.02840, 0.13383, 0.83777));\n"                                                         \
	"mat3 ACES1_output_mat = mat3(\n"                                                                    \
	"	vec3( 1.60475, -0.53108, -0.07367),\n"                                                       \
	"	vec3(-0.10208,  1.10813, -0.00605),\n"                                                       \
	"	vec3(-0.00327, -0.07276,  1.07602));\n"                                                      \
	"vec3 TONEMAP(vec3 x)\n"                                                                             \
	"{\n"                                                                                                \
	"	x *= ACES1_input_mat;\n"                                                                     \
	"	x = (x * (x + 0.0245786) - 0.000090537) / (x * (0.983729 * x + 0.4329510f) + 0.238081);\n"   \
	"	x *= ACES1_output_mat;\n"                                                                    \
	"	return clamp(x, 0.0, 1.0);\n"                                                                \
	"}\n"

static GLuint hdrpix__create_shader(GLenum type, const char* src)
{
	GLuint shader = glCreateShader(type); CHKGL;
	glShaderSource(shader, 1, &src, NULL); CHKGL;
	glCompileShader(shader); CHKGL;
	GLint status;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
	if (status == GL_FALSE) {
		GLint msglen;
		glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &msglen);
		GLchar* msg = (GLchar*) malloc(msglen + 1);
		assert(msg != NULL);
		glGetShaderInfoLog(shader, msglen, NULL, msg);
		const char* stype = type == GL_VERTEX_SHADER ? "VERTEX" : type == GL_FRAGMENT_SHADER ? "FRAGMENT" : "???";
		fprintf(stderr, "%s GLSL COMPILE ERROR: %s in\n\n%s\n", stype, msg, src);
		abort();
	}

	return shader;
}

static void hdrpix__prg_init(struct hdrpix__prg* prg, const char* vert_src, const char* frag_src, struct hdrpix__uniform* uniforms)
{
	GLuint vs = hdrpix__create_shader(GL_VERTEX_SHADER, vert_src);
	GLuint fs = hdrpix__create_shader(GL_FRAGMENT_SHADER, frag_src);

	GLuint program = glCreateProgram(); CHKGL;
	glAttachShader(program, vs); CHKGL;
	glAttachShader(program, fs); CHKGL;

	glLinkProgram(program); CHKGL;

	GLint status;
	glGetProgramiv(program, GL_LINK_STATUS, &status);
	if (status == GL_FALSE) {
		GLint msglen;
		glGetProgramiv(program, GL_INFO_LOG_LENGTH, &msglen);
		GLchar* msg = (GLchar*) malloc(msglen + 1);
		glGetProgramInfoLog(program, msglen, NULL, msg);
		fprintf(stderr, "shader link error: %s\n", msg);
		abort();
	}

	glDeleteShader(vs); CHKGL;
	glDeleteShader(fs); CHKGL;

	prg->program = program;
	int n_uniforms = 0;
	for (struct hdrpix__uniform* u = uniforms; u->name != NULL; u++, n_uniforms++) {}
	assert(n_uniforms <= HDRPIX__PRG_MAX_UNIFORMS);
	prg->n_uniforms = n_uniforms;
	memcpy(prg->uniforms, uniforms, n_uniforms * sizeof(uniforms[0]));
	for (int i = 0; i < n_uniforms; i++) {
		struct hdrpix__uniform* u = &prg->uniforms[i];
		u->location = glGetUniformLocation(program, u->name); CHKGL;
	}
}

static void hdrpix__prg_set_uniforms(struct hdrpix__prg* prg, void* data)
{
	for (int i = 0; i < prg->n_uniforms; i++) {
		struct hdrpix__uniform* u = &prg->uniforms[i];
		GLint loc = u->location;
		int n = u->element_count;
		if (loc < 0) continue;
		void* base = data + (size_t)u->offset;
		if (u->type == HDRPIX__FLOAT) {
			GLfloat* x = (GLfloat*)base;
			const int count = 1;
			switch (n) {
			case 1: glUniform1fv(loc, count, x); CHKGL; break;
			case 2: glUniform2fv(loc, count, x); CHKGL; break;
			case 3: glUniform3fv(loc, count, x); CHKGL; break;
			case 4: glUniform4fv(loc, count, x); CHKGL; break;
			default: assert(!"unhandled u->element_count");
			}
		} else if (u->type == HDRPIX__INT) {
			GLint* x = (GLint*)base;
			const int count = 1;
			switch (n) {
			case 1: glUniform1iv(loc, count, x); CHKGL; break;
			case 2: glUniform2iv(loc, count, x); CHKGL; break;
			case 3: glUniform3iv(loc, count, x); CHKGL; break;
			case 4: glUniform4iv(loc, count, x); CHKGL; break;
			default: assert(!"unhandled u->element_count");
			}
		} else {
			assert(!"unhandled u->type");
		}
	}
}

static void hdrpix__prg_use(struct hdrpix__prg* prg)
{
	glUseProgram(prg->program); CHKGL;
}

static void hdrpix__prg_free(struct hdrpix__prg* prg)
{
	glDeleteProgram(prg->program); CHKGL;
}

static inline unsigned char hdrpix_enc(float max_luminance, float value)
{
	return (unsigned char) fmaxf(0.0f, fminf(255.5f, (value / max_luminance) * 255.0f + 0.5f));
}

void hdrpix_init(struct hdrpix* hp, int config_width, int config_height, float max_luminance)
{
	memset(hp, 0, sizeof *hp);
	hp->config_width = config_width;
	hp->config_height = config_height;
	hp->max_luminance = max_luminance;

	{
		const char* vert_src =
		"#version 140\n"
		"\n"
		"uniform vec2 u_src_resolution;\n"
		"uniform vec2 u_d0;\n"
		"uniform vec2 u_d1;\n"
		"\n"
		"varying vec2 v_uv;\n"
		"\n"
		"void main(void)\n"
		"{\n"
		"\n"
		"	vec2 p;\n"
		"	vec2 uv0 = vec2(-0.5, -0.5);\n"
		"	vec2 uv1 = uv0 + u_src_resolution;\n"
		"\n"
		"	if        (" HDRPIX__IS_Q0 ") {\n"
		"		p = vec2(u_d0.x, u_d0.y);\n"
		"		v_uv = vec2(uv0.x, uv1.y);\n"
		"	} else if (" HDRPIX__IS_Q1 ") {\n"
		"		p = vec2(u_d1.x, u_d0.y);\n"
		"		v_uv = vec2(uv1.x, uv1.y);\n"
		"	} else if (" HDRPIX__IS_Q2 ") {\n"
		"		p = vec2(u_d1.x, u_d1.y);\n"
		"		v_uv = vec2(uv1.x, uv0.y);\n"
		"	} else if (" HDRPIX__IS_Q3 ") {\n"
		"		p = vec2(u_d0.x, u_d1.y);\n"
		"		v_uv = vec2(uv0.x, uv0.y);\n"
		"	}\n"
		"	gl_Position = vec4(p, 0.0, 1.0);\n"
		"}\n"
		;

		const char* frag_src =
		"#version 140\n"
		"\n"
		"uniform vec2 u_src_resolution;\n"
		"uniform sampler2D u_src_texture;\n"
		"uniform float u_scale;\n"
		"\n"
		"varying vec2 v_uv;\n"
		"\n"
		"void main(void)\n"
		"{\n"
		"	vec2 uv = floor(v_uv) + 0.5;\n"
		"	uv += 1.0 - clamp((1.0 - fract(v_uv)) * u_scale, 0.0, 1.0);\n"
		"	gl_FragColor = texture2D(u_src_texture, uv / u_src_resolution);\n"
		"}\n"
		;

		hdrpix__prg_init(&hp->pix_prg, vert_src, frag_src, (struct hdrpix__uniform[]) {
			HDRPIX__UNIFORM_FLOATS(struct hdrpix__pix_uniforms, u_src_resolution) ,
			HDRPIX__UNIFORM_INTS(  struct hdrpix__pix_uniforms, u_src_texture)    ,
			HDRPIX__UNIFORM_FLOATS(struct hdrpix__pix_uniforms, u_d0)             ,
			HDRPIX__UNIFORM_FLOATS(struct hdrpix__pix_uniforms, u_d1)             ,
			HDRPIX__UNIFORM_FLOATS(struct hdrpix__pix_uniforms, u_scale)          ,
			{0},
		});
	}
}

void hdrpix_set_display_dimensions(struct hdrpix* hp, int width, int height)
{
	if (hp->display_width == width && hp->display_height == height) return;
	hp->display_width = width;
	hp->display_height = height;
	if (width == 0 || height == 0) return;

	int canvas_width, canvas_height;
	if (hp->config_width > 0 && hp->config_height == 0) {
		canvas_width = hp->config_width;
		canvas_height = (canvas_width * height) / width;
	} else if (hp->config_width == 0 && hp->config_height > 0) {
		canvas_height = hp->config_height;
		canvas_width = (canvas_height * width) / height;
	} else if (hp->config_width > 0 && hp->config_height > 0) {
		canvas_width = hp->config_width;
		canvas_height = hp->config_height;
	} else {
		assert(!"INVALID CONFIG");
	}

	if (hp->canvas_width == canvas_width && hp->canvas_height == canvas_height) {
		return;
	}

	hp->canvas_width = canvas_width;
	hp->canvas_height = canvas_height;
	hp->canvas = realloc(hp->canvas, HDRPIX_NCC * canvas_width * canvas_height);
}

static void hdrpix__draw_quad(void)
{
	glDrawArrays(GL_TRIANGLES, 0, 6); CHKGL;
}

void hdrpix_present(struct hdrpix* hp, struct hdrpix_shader_config* cfg)
{
	hdrpix__tex_setup(&hp->canvas_tex, hp->canvas_width, hp->canvas_height);
	assert((hp->canvas_tex.width == hp->canvas_width) && (hp->canvas_tex.height == hp->canvas_height));
	hdrpix__tex_upload(&hp->canvas_tex, hp->canvas);

	if (hp->current_shader_config.shader != cfg->shader) {
		// release
		switch (hp->current_shader_config.shader) {
		case HDRPIX_SHADER_NULL: break;
		case HDRPIX_SHADER_NOISY: {
			hdrpix__fb_free(&hp->ss.noisy.fb);
			hdrpix__prg_free(&hp->ss.noisy.prg);
		} break;
		default: assert(!"WHAT");
		}

		// present
		switch (cfg->shader) {
		case HDRPIX_SHADER_NOISY: {
			const char* vert_src =
			"#version 140\n"
			"varying vec2 v_uv;\n"
			"void main(void)\n"
			"{\n"
			"	if        (" HDRPIX__IS_Q0 ") {\n"
			"		gl_Position = vec4( -1, -1, 0,1);\n"
			"		v_uv = vec2(0,0);\n"
			"	} else if (" HDRPIX__IS_Q1 ") {\n"
			"               gl_Position = vec4(  1, -1, 0,1);\n"
			"		v_uv = vec2(1,0);\n"
			"	} else if (" HDRPIX__IS_Q2 ") {\n"
			"               gl_Position = vec4(  1,  1, 0,1);\n"
			"		v_uv = vec2(1,1);\n"
			"	} else if (" HDRPIX__IS_Q3 ") {\n"
			"               gl_Position = vec4( -1,  1, 0,1);\n"
			"		v_uv = vec2(0,1);\n"
			"	}\n"
			"}\n"
			;
			const char* frag_src =
			"#version 140\n"
			HDRPIX__TONEMAP_ACES1
			"uniform sampler2D u_src_texture;\n"
			"uniform vec2 u_src_scale;\n"
			"uniform float u_max_luminance;\n"
			"uniform float u_seed;\n"
			"varying vec2 v_uv;\n"
			"float rand(vec2 co)\n"
			"{\n"
			"	return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);\n"
			"}\n"
			"float rnd(int i0, int i1)\n"
			"{\n"
			"	vec2 xy = vec2(i0,i1);\n"
			"	xy += v_uv * -12.34 * u_seed;\n"
			"	return rand(xy);\n"
			"}\n"
			"void main(void)\n"
			"{\n"
			"	vec3 c = texture2D(u_src_texture, v_uv).xyz;\n"
			"	const int N=50;\n"
			"	const float ooN = 1.0 / float(N);\n"
			"	const float R=40;\n"
			"	for (int i = 0; i < N; i++) {\n"
			"		float r = sqrt(rnd(i,0));\n"
			"		float theta = rnd(i,1) * 6.283185307179586;\n"
			"		vec2 uv2 = v_uv + u_src_scale*R*r*vec2(cos(theta), sin(theta));\n"
			"		float scalar = exp(-(r*r*3.0)) * ooN;\n"
			"		c += texture2D(u_src_texture, uv2).xyz * scalar;\n"
			"	}\n"
			"	c *= u_max_luminance;\n"
			"	c = TONEMAP(c);\n"
			"	gl_FragColor = vec4(c,1);\n"
			"}\n"
			;
			hdrpix__prg_init(&hp->ss.noisy.prg, vert_src, frag_src, (struct hdrpix__uniform[]) {
				HDRPIX__UNIFORM_INTS(  struct hdrpix__noisy_uniforms, u_src_texture)    ,
				HDRPIX__UNIFORM_FLOATS(struct hdrpix__noisy_uniforms, u_src_scale)      ,
				HDRPIX__UNIFORM_FLOATS(struct hdrpix__noisy_uniforms, u_max_luminance)  ,
				HDRPIX__UNIFORM_FLOATS(struct hdrpix__noisy_uniforms, u_seed)           ,
				{0},
			});


		} break;
		default: assert(!"WHAT");
		}

		hp->current_shader_config.shader = cfg->shader;
	}

	// present
	GLuint present_texture = -1;
	switch (hp->current_shader_config.shader) {
	case HDRPIX_SHADER_NOISY: {
		hdrpix__fb_bind(&hp->ss.noisy.fb, hp->canvas_width, hp->canvas_height);
		hdrpix__prg_use(&hp->ss.noisy.prg);
		glBindTexture(GL_TEXTURE_2D, hp->canvas_tex.texture); CHKGL;
		hdrpix__prg_set_uniforms(&hp->ss.noisy.prg, &(struct hdrpix__noisy_uniforms) {
			.u_src_texture = 0,
			.u_src_scale = { 1.0f / (float)hp->canvas_width, 1.0f / (float)hp->canvas_height},
			.u_max_luminance = hp->max_luminance,
			.u_seed = (float)rand() / (float)RAND_MAX,
		});
		hdrpix__draw_quad();
		present_texture = hp->ss.noisy.fb.tex.texture;
	} break;
	default: assert(!"WHAT");
	}

	glBindFramebuffer(GL_FRAMEBUFFER, 0); CHKGL;
	glViewport(0, 0, hp->display_width, hp->display_height);

	glClearColor(0,0,0,0);
	glClear(GL_COLOR_BUFFER_BIT);

	hdrpix__prg_use(&hp->pix_prg);
	assert(present_texture != -1);
	glBindTexture(GL_TEXTURE_2D, present_texture); CHKGL;
	{
		const float src_width = hp->canvas_width;
		const float src_height = hp->canvas_height;
		const float dst_width  = hp->display_width;
		const float dst_height = hp->display_height;
		const float src_aspect = src_width / src_height;
		const float dst_aspect = dst_width / dst_height;
		float dx0, dy0, dx1, dy1, scale;
		if (src_aspect > dst_aspect) {
			dx0 = -1.0f;
			dx1 =  1.0f;
			scale = dst_width / src_width;
			const float margin_norm = (dst_height - src_height*scale) / dst_height;
			dy0 = -1.0f + margin_norm;
			dy1 =  1.0f - margin_norm;
		} else {
			dy0 = -1.0f;
			dy1 =  1.0f;
			scale = dst_height / src_height;
			const float margin_norm = (dst_width - src_width*scale) / dst_width;
			dx0 = -1.0f + margin_norm;
			dx1 =  1.0f - margin_norm;
		}
		hdrpix__prg_set_uniforms(&hp->pix_prg, &(struct hdrpix__pix_uniforms) {
			.u_src_resolution = {src_width, src_height},
			.u_d0 = {dx0, dy0},
			.u_d1 = {dx1, dy1},
			.u_scale = scale,
			.u_src_texture = 0,
		});
	}

	hdrpix__draw_quad();
}

#define HDRPIX_H
#endif
