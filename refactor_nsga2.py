import re

with open('include/nsga2.hpp', 'r') as f:
    content = f.read()

# 1. Template Individual
content = content.replace("struct Individual {", "template <typename ManifoldType>\nstruct Individual {")
content = content.replace("FourierManifold manifold;", "ManifoldType manifold;")
content = content.replace("bool dominates(const Individual& o) const {", "bool dominates(const Individual& o) const {") # Inside struct, Individual is fine

# 2. Template NSGA2
content = content.replace("class NSGA2 {", "template <typename ManifoldType>\nclass NSGA2 {")
content = content.replace("ObjFn obj_fn", "ObjFn<ManifoldType> obj_fn")
content = content.replace("ObjFn      obj_fn_;", "ObjFn<ManifoldType>      obj_fn_;")

# 3. Replace std::vector<Individual> with std::vector<Individual<ManifoldType>>
content = content.replace("std::vector<Individual>", "std::vector<Individual<ManifoldType>>")
content = content.replace("const Individual&", "const Individual<ManifoldType>&")
content = content.replace("Individual best_accuracy()", "Individual<ManifoldType> best_accuracy()")
content = content.replace("Individual knee_point()", "Individual<ManifoldType> knee_point()")

# 4. Replace FourierManifold inside NSGA2 methods
content = content.replace("ind.manifold = FourierManifold(dim_, n_harm);", "ind.manifold = ManifoldType(dim_, n_harm);")
content = content.replace("FourierManifold m1, m2;", "ManifoldType m1, m2;")

# 5. Template FMClassifier -> Classifier
content = content.replace("class FMClassifier {", "template <typename ManifoldType>\nclass Classifier {")
content = content.replace("FMClassifier(NSGAConfig", "Classifier(NSGAConfig")
content = content.replace("NSGA2 engine", "NSGA2<ManifoldType> engine")
content = content.replace("std::vector<FourierManifold> manifolds_;", "std::vector<ManifoldType> manifolds_;")
content = content.replace("const std::vector<FourierManifold>& manifolds() const", "const std::vector<ManifoldType>& manifolds() const")

with open('include/nsga2.hpp', 'w') as f:
    f.write(content)

print("Refactored nsga2.hpp")
