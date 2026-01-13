export Material,
    Light,
    Dark,
    NMAT


@enum Material begin
    Light = 1
    Dark  = 2
end

const NMAT = length(instances(Material))

function Base.convert(::Type{Material}, mat::String)
    Material(mat)
end

function Material(mat::String)
    v = if mat == "Light"
        1
    elseif mat == "Dark"
        2
    else
        error("Material $(mat) is not valid")
    end
    Material(v)
end

import Base.show

Base.show(io::IO, x::Material) = x == Light ? "light" : "dark"
