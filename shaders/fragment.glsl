#version 330 core
in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D screenTexture;
void main() {
    vec3 color = texture(screenTexture, TexCoord).rgb;
    FragColor = vec4(color, 1.0);  // инверсия цветов
}
