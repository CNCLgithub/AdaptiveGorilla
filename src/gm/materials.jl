export Material,
    Light,
    Dark,
    NMAT


@enum Material begin
    Light # 1
    Dark  # 2
end

# REVIEW: needed?
# const materials = collect(instances(Material))

NMAT = length(instances(Material))

import Base.show

Base.show(io::IO, x::Material) = x == Light ? "light" : "dark"
