import re

with open('FM_NSGA_II_Animation.py', 'r') as f:
    content = f.read()

# Extraer helpers 3D (494-533)
helpers_match = re.search(r'(# ─── Datos 3D sintéticos y Helpers ────────────────────────────────────────────.*?)(?=        # ── 9\.5 Escena 3D ─────────────────────────────────────────────────────────)', content, re.DOTALL)
helpers_code = helpers_match.group(1)

# Extraer el método _scene_3d (535-578)
scene_match = re.search(r'(        # ── 9\.5 Escena 3D ─────────────────────────────────────────────────────────.*)', content, re.DOTALL)
scene_code = scene_match.group(1)

# Eliminar helpers y scene_3d del final del archivo
content = content.replace(helpers_code, '')
content = content.replace(scene_code, '')

# Arreglar la indentación de scene_code (está mal indentado)
# En lugar de arreglar con regex, vamos a reformatearlo:
scene_code_fixed = "    # ── 9.5 Escena 3D ─────────────────────────────────────────────────────────\n"
for line in scene_code.split('\n')[1:]:
    if line.startswith('        def '):
        scene_code_fixed += '    def ' + line[12:] + '\n'
    elif line.startswith('            '):
        scene_code_fixed += '        ' + line[12:] + '\n'
    elif line.startswith('    def '):
        scene_code_fixed += line + '\n'
    else:
        scene_code_fixed += line + '\n'

# Insertar helpers antes de la clase (antes de # ══════)
class_header = '# ══════════════════════════════════════════════════════════════════════════════\nclass FMNSGAFullAnimation(ThreeDScene):'
content = content.replace(class_header, helpers_code + '\n' + class_header)

# Insertar _scene_3d antes de _outro
outro_header = '    # ── 10. Outro ─────────────────────────────────────────────────────────────'
content = content.replace(outro_header, scene_code_fixed + '\n' + outro_header)

with open('FM_NSGA_II_Animation.py', 'w') as f:
    f.write(content)

print("Estructura corregida.")
