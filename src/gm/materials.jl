export Material,
    NMAT


@enum Material begin
    Light # 1
    Dark  # 2
end



NMAT = length(instances(Material))
