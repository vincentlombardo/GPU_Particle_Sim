#pragma once
#include <glad/glad.h>

// Read-only consumer of RenderState (as a raw GL buffer handle + count).
// Knows nothing about SimulationState, CUDA, or MPI.
class Renderer
{
public:
    void initialize();
    void render(GLuint positionVBO, int count);
    void shutdown();

private:
    GLuint program = 0;
    GLuint vao     = 0;

    GLuint compileShader(GLenum type, const char* path);
};
