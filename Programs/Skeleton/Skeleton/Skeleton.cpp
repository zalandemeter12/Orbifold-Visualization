//=============================================================================================
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Demeter Zalán
// Neptun : VERF1U
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

struct Material {
	vec3 ka, kd, ks;
	float  shininess;
	vec3 F0;
	int rough, reflective;
	int portal = false;
};

struct RoughMaterial : Material {
	RoughMaterial(vec3 _kd, vec3 _ks, float _shininess) {
		ka = _kd * M_PI;
		kd = _kd;
		ks = _ks;
		shininess = _shininess;
		rough = true;
		reflective = false;
	}
};

inline vec3 operator/(const vec3& v1, const vec3& v2) { return vec3(v1.x / v2.x, v1.y / v2.y, v1.z / v2.z); }

struct SmoothMaterial : Material {
	SmoothMaterial(vec3 n, vec3 kappa) {
		vec3 one(1, 1, 1);
		F0 = ((n - one) * (n - one) + kappa * kappa) / ((n + one) * (n + one) + kappa * kappa);
		rough = false;
		reflective = true;
	}
};

struct Hit {
	float t;
	vec3 position, normal;
	Material* material;
	Hit() { t = -1; }
};

struct Ray {
	vec3 start, dir;
	Ray(vec3 _start = vec3(), vec3 _dir = vec3()) { start = _start; dir = normalize(_dir); }
};

class Intersectable {
protected:
	Material* material;
public:
	virtual Hit intersect(const Ray& ray) = 0;
};

struct Paraboloid : public Intersectable {
	float a, b, c;

	Paraboloid(float _a, float _b, float _c, Material* _material) {
		a = _a; b = _b; c = _c;
		material = _material;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		float A = a * (ray.dir.x * ray.dir.x) + b * (ray.dir.y * ray.dir.y);
		float B = 2.0f * a * ray.start.x * ray.dir.x + 2.0f * b * ray.start.y * ray.dir.y - c * ray.dir.z;
		float C = a * (ray.start.x * ray.start.x) + b * (ray.start.y * ray.start.y) - c * ray.start.z;
		float discr = B * B - 4.0f * A * C;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-B + sqrt_discr) / 2.0f / A;
		float t2 = (-B - sqrt_discr) / 2.0f / A;
		if (t1 <= 0) return hit;
		vec3 intersect1 = ray.start + ray.dir * t1;
		vec3 intersect2 = ray.start + ray.dir * t2;
		if (powf((intersect2.x), 2) + powf((intersect2.y), 2) + powf((intersect2.z), 2) <= 0.3 * 0.3) {
			hit.t = t2;
			hit.position = intersect2;
		}
		else if (powf((intersect1.x), 2) + powf((intersect1.y), 2) + powf((intersect1.z), 2) <= 0.3 * 0.3) {
			hit.t = t1;
			hit.position = intersect1;
		}
		else return hit;
		vec3 u = vec3(1, 0, (2 * a * hit.position.x) / c);
		vec3 w = vec3(0, 1, (2 * b * hit.position.y) / c);
		vec3 d = normalize(cross(u, w));
		hit.normal = hit.position - (hit.position + d);
		hit.material = material;
		return hit;
	}
};

struct Dodecahedron : public Intersectable {
	struct face5 {
		unsigned int i, j, k, l, m;
		face5(unsigned int _i = 0, unsigned int _j = 0, unsigned int _k = 0, unsigned int _l = 0, unsigned int _m = 0)
			: i(_i - 1), j(_j - 1), k(_k - 1), l(_l - 1), m(_m - 1) {}
	};

	std::vector<vec3> vertices;
	std::vector<face5> faces;
	Material* materialRefl;

	Dodecahedron(const std::vector<vec3>& _vertices, const std::vector<face5>& _faces, Material* _material, Material* _materialRefl) {
		vertices = _vertices;
		faces = _faces;
		material = _material;
		materialRefl = _materialRefl;
	}

	float getDist(Hit hit, vec3 A, vec3 B) {
		return length(cross((hit.position - A), (hit.position - B))) / length(B - A);
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		face5 currentFace;
		for (const face5& face: faces) {
			vec3 p1 = vertices[face.i];
			vec3 p2 = vertices[face.j];
			vec3 p3 = vertices[face.k];
			vec3 u = p2 - p1;
			vec3 v = p3 - p1;
			vec3 normal = normalize(cross(u, v));
			float t = dot(normal,(p1 - ray.start)) / dot(normal, ray.dir);
			if ((t > 0 && hit.t == -1) || (t > 0 && t < hit.t)) {
				hit.t = t;
				hit.position = ray.start + ray.dir * hit.t;
				hit.normal = normal;
				hit.material = material;
				currentFace = face;
			}
		}
		vec3 A = vertices[currentFace.i];
		vec3 B = vertices[currentFace.j];
		vec3 C = vertices[currentFace.k];
		vec3 D = vertices[currentFace.l];
		vec3 E = vertices[currentFace.m];

		float d1 = getDist(hit, A, B);
		float d2 = getDist(hit, B, C);
		float d3 = getDist(hit, C, D);
		float d4 = getDist(hit, D, E);
		float d5 = getDist(hit, E, A);

		if (d1 < 0.1 || d2 < 0.1 || d3 < 0.1 || d4 < 0.1 || d5 < 0.1)
			hit.material = material;
		else
			hit.material = materialRefl;

		return hit;
	}
};

struct Camera {
	vec3 eye, lookat, right, up;
	float fov;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float _fov) {
		eye = _eye;
		lookat = _lookat;
		fov = _fov;
		vec3 w = eye - lookat;
		float f = length(w);
		right = normalize(cross(vup, w)) * f * tanf(fov / 2);
		up = normalize(cross(w, right)) * f * tanf(fov / 2);
	}

	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0 * (X + 0.5) / windowWidth - 1) + up * (2.0 * (Y + 0.5) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}
	
	void Animate(float dt) {
		eye = vec3((eye.x - lookat.x) * cosf(dt) + (eye.z - lookat.z) * sinf(dt) + lookat.x,
			eye.y,
			-(eye.x - lookat.x) * sinf(dt) + (eye.z - lookat.z) * cosf(dt) + lookat.z);
		set(eye, lookat, up, fov);
	}
};

struct Light {
	vec3 position;
	vec3 color;
	Light(vec3 _position = vec3(), vec3 _color = vec3(1, 1, 1)) {
		position = _position;
		color = _color;
	}
	vec3 radiance(float distance) {
		return color / (powf(distance, 2));
	}
};

const float epsilon = 0.0001;
const int maxdepth = 5;

class Scene {
	std::vector<Intersectable*> objects;
	std::vector<Light*> lights;
	Camera camera;
	vec3 La;
public:
	void build() {
		vec3 eye = vec3(0, 0, 1.35), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		La = vec3(0.75, 0.75, 0.75);
		lights.push_back(new Light(vec3(0, 0.5, 0), vec3(1, 1, 1)));
		
		Material* gold = new SmoothMaterial(vec3(0.17, 0.35, 1.5), vec3(3.1, 2.7, 1.9));
		objects.push_back(new Paraboloid(5, 5, 1, gold));
		
		vec3 vertexArr[] = {vec3(0,0.618,1.618), vec3(0,-0.618,1.618),	vec3(0,-0.618,-1.618),	vec3(0,0.618,-1.618), 
							vec3(1.618,0,0.618), vec3(-1.618,0,0.618),	vec3(-1.618,0,-0.618),	vec3(1.618,0,-0.618),
							vec3(0.618,1.618,0), vec3(-0.618,1.618,0),	vec3(-0.618,-1.618,0),	vec3(0.618,-1.618,0),
							vec3(1,1,1),		 vec3(-1,1,1),			vec3(-1,-1,1),			vec3(1,-1,1),
							vec3(1,-1,-1),		 vec3(1,1,-1),			vec3(-1,1,-1),			vec3(-1,-1,-1) };
		std::vector<vec3> vertices(vertexArr, vertexArr + 20);

		Dodecahedron::face5 faceArr[] = {Dodecahedron::face5(1,2,16,5,13),		Dodecahedron::face5(1,13,9,10,14),
										 Dodecahedron::face5(1,14,6,15,2),		Dodecahedron::face5(2,15,11,12,16),
										 Dodecahedron::face5(3,4,18,8,17),		Dodecahedron::face5(3,17,12,11,20),
										 Dodecahedron::face5(3,20,7,19,4),		Dodecahedron::face5(19,10,9,18,4),
										 Dodecahedron::face5(16,12,17,8,5),		Dodecahedron::face5(5,8,18,9,13),
										 Dodecahedron::face5(14,10,19,7,6),		Dodecahedron::face5(6,7,20,11,15) };
		std::vector<Dodecahedron::face5> faces(faceArr, faceArr + 12);

		Material* ceramicPink = new RoughMaterial(vec3(0.3, 0.25, 0.3), vec3(2, 2, 2), 1000);
		Material* zinc = new SmoothMaterial(vec3(1.2338, 0.92943, 0.67767), vec3(5.8730, 4.9751, 4.0122));
		zinc->portal = true;
		objects.push_back(new Dodecahedron(vertices, faces, ceramicPink, zinc));
	}

	void render(std::vector<vec4>& image) {
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y), 0);
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable* object : objects) {
			Hit hit = object->intersect(ray);
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	vec3 reflect(vec3 V, vec3 N) {
		return V - N * dot(N, V) * 2;
	};

	vec3 Fresnel(vec3 V, vec3 N, vec3 F0) {
		float cosa = -dot(V, N);
		return F0 + (vec3(1, 1, 1) - F0) * powf(1 - cosa, 5);
	}

	vec3 trace(Ray ray, int d = 0) {
		if (d > maxdepth) return La;
		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return La;
		vec3 outRadiance = vec3(0, 0, 0);

		if (hit.material->rough) {
			outRadiance = hit.material->ka * La;
			for (Light* light : lights) {
				vec3 direction = normalize(light->position - hit.position);
				float distance = length(light->position - hit.position);
				vec3 Le = light->radiance(distance);
				Ray shadowRay(hit.position + hit.normal * epsilon, direction);
				Hit shadowHit = firstIntersect(shadowRay);
				float cosTheta = dot(hit.normal, direction);
				if (cosTheta > 0 && (shadowHit.t < 0 || shadowHit.t > distance)) {
					outRadiance = outRadiance + Le * hit.material->kd * cosTheta;
					vec3 halfway = normalize(-ray.dir + direction);
					float cosDelta = dot(hit.normal, halfway);
					if (cosDelta > 0) outRadiance = outRadiance + Le * hit.material->ks * powf(cosDelta, hit.material->shininess);
				}
			}
		}

		if (hit.material->reflective) {
			vec3 reflectedDir = ray.dir - hit.normal * dot(hit.normal, ray.dir) * 2.0f;
			vec3 reflectedPoint = hit.position + hit.normal * epsilon;
			float cosa = -dot(ray.dir, hit.normal);
			vec3 one(1, 1, 1);
			vec3 F = hit.material->F0 + (one - hit.material->F0) * powf(1 - cosa, 5);
			if (hit.material->portal) {
				float theta = 72 * M_PI / 180;
				vec3 rotDir = reflectedDir * cosf(theta) + cross(hit.normal, reflectedDir) * sinf(theta) + hit.normal * dot(hit.normal, reflectedDir) * (1 - cosf(theta));
				vec3 rotPoint = reflectedPoint * cosf(theta) + cross(hit.normal, reflectedPoint) * sinf(theta) + hit.normal * dot(hit.normal, reflectedPoint) * (1 - cosf(theta));
				outRadiance = outRadiance + F * trace(Ray(rotPoint, rotDir), d + 1);
			}
			else {
				outRadiance = outRadiance + F * trace(Ray(reflectedPoint, reflectedDir), d + 1);
			}
		}
		
		return outRadiance;
	}

	void Animate(float dt) { camera.Animate(dt); }
};

GPUProgram gpuProgram;
Scene scene;

const char* vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;				
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 	
	}
)";

const char* fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			
	out vec4 fragmentColor;		

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";

class FullScreenTexturedQuad {
	unsigned int vao;
	Texture texture;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image)
		: texture(windowWidth, windowHeight, image) {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		unsigned int vbo;		
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo); 
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     
	}

	void Draw() {
		glBindVertexArray(vao);
		gpuProgram.setUniform(texture, "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	}
};

FullScreenTexturedQuad* fullScreenTexturedQuad;

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();
	scene.Animate(0.2f);
	std::vector<vec4> image(windowWidth * windowHeight);
	long timeStart = glutGet(GLUT_ELAPSED_TIME);
	scene.render(image);
	long timeEnd = glutGet(GLUT_ELAPSED_TIME);
	printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

void onDisplay() {
	glClearColor(1.0f, 0.5f, 0.8f, 1.0f); 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	std::vector<vec4> image(windowWidth * windowHeight);
	scene.render(image);
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) {}
void onKeyboardUp(unsigned char key, int pX, int pY) {}
void onMouse(int button, int state, int pX, int pY) {}
void onMouseMotion(int pX, int pY) {}

void onIdle() {
	scene.Animate(0.05f);
	glutPostRedisplay();
}