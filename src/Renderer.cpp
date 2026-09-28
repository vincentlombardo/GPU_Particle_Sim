#include "Renderer.h"
#include <fstream>
#include <sstream>
#include <string>
#include <cstdio>
#include <cstdlib>

namespace
{
    std::string readFile(const char* path)
    {
        std::ifstream file(path);
        if (!file)
        {
            std::fprintf(stderr, "Could not open shader file: %s\n", path);
            std::exit(EXIT_FAILURE);
        }
        std::stringstream ss;
        ss << file.rdbuf();
        return ss.str();
    }

    void checkShaderCompile(GLuint shader, const char* label)
    {
        GLint success = 0;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success)
        {
            char log[1024];
            glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
            std::fprintf(stderr, "Shader compile error (%s): %s\n", label, log);
            std::exit(EXIT_FAILURE);
        }
    }

    void checkProgramLink(GLuint program)
    {
        GLint success = 0;
        glGetProgramiv(program, GL_LINK_STATUS, &success);
        if (!success)
        {
            char log[1024];
            glGetProgramInfoLog(program, sizeof(log), nullptr, log);
            std::fprintf(stderr, "Program link error: %s\n", log);
            std::exit(EXIT_FAILURE);
        }
    }
}

GLuint Renderer::compileShader(GLenum type, const char* path)
{
    std::string src = readFile(path);
    const char* srcPtr = src.c_str();

    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &srcPtr, nullptr);
    glCompileShader(shader);
    checkShaderCompile(shader, path);
    return shader;
}

void Renderer::initialize()
{
    GLuint vert = compileShader(GL_VERTEX_SHADER,   "shaders/particle.vert");
    GLuint frag = compileShader(GL_FRAGMENT_SHADER, "shaders/particle.frag");

    program = glCreateProgram();
    glAttachShader(program, vert);
    glAttachShader(program, frag);
    glLinkProgram(program);
    checkProgramLink(program);

    glDeleteShader(vert);
    glDeleteShader(frag);

    glGenVertexArrays(1, &vao);

    glEnable(GL_PROGRAM_POINT_SIZE);
}

void Renderer::render(GLuint positionVBO, int count)
{
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(program);
    glBindVertexArray(vao);

    glBindBuffer(GL_ARRAY_BUFFER, positionVBO);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glDrawArrays(GL_POINTS, 0, count);

    glBindVertexArray(0);
    glUseProgram(0);
}

void Renderer::shutdown()
{
    if (vao)     glDeleteVertexArrays(1, &vao);
    if (program) glDeleteProgram(program);
    vao = 0;
    program = 0;
}
