#################################################################################
# Implements solution to Problem A in https://arxiv.org/abs/2105.07472          #
# - aided by Grok:                                                              #
#     https://grok.com/share/c2hhcmQtMg_6011361d-4a17-4731-b776-6a4f230fbd7c    #
#################################################################################

export bell_idx_to_gstring, growthstring_to_uint, uint_to_growthstring

function bell_number(n::Int)
    if n == 0
        return 1
    end
    
    # Create Bell triangle
    bell = zeros(Int, n + 1, n + 1)
    bell[1, 1] = 1
    
    for i in 2:n+1
        # First element in each row is the last element of previous row
        bell[i, 1] = bell[i-1, i-1]
        
        # Fill the rest of the row
        for j in 2:i
            bell[i, j] = bell[i, j-1] + bell[i-1, j-1]
        end
    end
    
    return bell[n+1, 1]
end

# # Function to compute the Bell number for n elements
# function bell_number(n::Int)::Int
#     if n == 0
#         return 1
#     end
#     bell = zeros(Int, n + 1)
#     bell[1] = 1  # B_0 = 1
#     for i = 1:n
#         bell[i + 1] = sum(binomial(i, k) * bell[k + 1] for k = 0:i-1)
#     end
#     return bell[n + 1]
# end

# Helper function to build a binary merge tree from a sorted list of elements
function build_tree(elems::Vector{Int})::Any
    if length(elems) == 1
        return elems[1]
    end
    foldl((x, y) -> (x, y), elems)
end

# Helper function to flatten the tree and get sorted elements in the block
function flatten_tree(t::Any)::Vector{Int}
    if t isa Integer
        return [t]
    elseif t isa Tuple
        return sort!(vcat(flatten_tree(t[1]), flatten_tree(t[2])))
    else
        error("Invalid tree structure")
    end
end

# Function to convert restricted growth string a (0-based vector) to the partition representation
# Groups elements by labels, builds trees, sorts blocks by minimum element
function growth_to_partition(a::Vector{Int}, n::Int)::Vector{Any}
    blocks = Dict{Int, Vector{Int}}()
    for i = 1:n
        label = a[i]
        if !haskey(blocks, label)
            blocks[label] = Vector{Int}()
        end
        push!(blocks[label], i)
    end
    reps = Vector{Any}()
    for (_, blk) in blocks
        sorted_blk = sort!(blk)
        tree = build_tree(sorted_blk)
        push!(reps, tree)
    end
    # Sort the blocks by the minimum element in each
    sorted_reps = sort(reps; by = t -> flatten_tree(t)[1])
    return sorted_reps
end

# Implements Algorithm V from the manuscript (Section 3) to advance to the next partition
# Modifies a and b in place
# Returns true if advanced, false if end of enumeration
function next_partition!(a::Vector{Int}, b::Vector{Int}, n::Int)::Bool
    c = n
    while c > 1 && (a[c] == n - 1 || a[c] > b[c])
        c -= 1
    end
    if c == 1
        return false  # End of enumeration
    end
    a[c] += 1
    for i = c + 1:n
        a[i] = 0
        b[i] = max(a[i - 1], b[i - 1])
    end
    return true
end

# Main function: Maps index to partition by iteratively advancing from the first partition
# Uses constant memory, advances in constant amortized time per call
function bell_idx_to_gstring(n::Int, idx::Int)::Vector{Int}
    Bn = bell_number(n)
    if idx < 1 || idx > Bn
        error("Index $(idx) for S($(n)) must be [1,  $Bn]")
    end
    # Initialize to the first partition: all zeros (all elements in one block)
    a = zeros(Int, n)
    b = zeros(Int, n)
    # Advance idx-1 times to reach the idx-th partition
    for _ = 2:idx
        if !next_partition!(a, b, n)
            error("Unexpected end of enumeration")
        end
    end
    return a
end

# function growthstring_to_uint(a::Vector{Int})::UInt
#     n = length(a)
#     @assert n ≥ 1 && a[1] == 0  # 1-based convention: a[1]=0

#     result = UInt(0)
#     current_max = 0

#     for i = 1:n
#         digit = a[i]
#         @assert 0 ≤ digit ≤ current_max + 1

#         # At step i, we have (current_max + 2) possible values: 0 .. current_max+1
#         result = result * (current_max + 2) + digit

#         current_max = max(current_max, digit)
#     end

#     return result
# end

# function uint_to_growthstring(num::UInt, n::Int)::Vector{Int}
#     if n == 0
#         return Int[]
#     end

#     a = zeros(Int, n)
#     current_max = 0
#     remaining = num

#     for i = n:-1:1
#         base = current_max + 2
#         digit = remaining % base
#         a[i] = digit
#         remaining ÷= base

#         current_max = max(current_max, digit)
#     end

#     # The first element must be 0
#     @assert a[1] == 0 "Invalid encoding for restricted growth string"

#     return a
# end

using Base: foldl

function compute_f(n::Int)::Matrix{BigInt}
    f = zeros(BigInt, n + 1, n + 2)  # Rows: m=0 to n, columns: cm=0 to n+1 (extra for safety)
    for cm = 0:n
        f[1, cm + 1] = 1
    end
    for m = 1:n
        for cm = 0:n
            s = BigInt(0)
            for d = 0:cm + 1
                new_cm = max(cm, d)
                if new_cm <= n
                    s += f[m, new_cm + 1]
                end
            end
            f[m + 1, cm + 1] = s
        end
    end
    return f
end

function growthstring_to_uint(gs::Vector{Int})::UInt
    n = length(gs)
    if n == 0
        return UInt(0)
    end
    @assert gs[1] == 0 "First element must be 0"
    f = compute_f(n)
    rank = BigInt(0)
    cm = 0
    for j = 2:n
        @assert gs[j] <= cm + 1 "Invalid restricted growth string"
        k = n - j  # Remaining positions after this one
        for d = 0:gs[j] - 1
            new_cm = max(cm, d)
            sub_count = (new_cm <= n) ? f[k + 1, new_cm + 1] : BigInt(0)
            rank += sub_count
        end
        cm = max(cm, gs[j])
    end
    if rank > typemax(UInt)
        error("Rank exceeds UInt maximum")
    end
    return UInt(rank)
end

function uint_to_growthstring(num::UInt, n::Int)::Vector{Int}
    if n == 0
        return Int[]
    end
    f = compute_f(n)
    total = f[n, 1]  # B_n
    r = BigInt(num)
    if r >= total
        error("Number out of range: must be between 0 and Bell(n)-1")
    end
    a = zeros(Int, n)
    a[1] = 0
    cm = 0
    remaining = n - 1
    for pos = 2:n
        cum = BigInt(0)
        max_d = cm + 1
        found = false
        for d = 0:max_d
            new_cm = max(cm, d)
            sub_count = (new_cm <= n) ? f[remaining, new_cm + 1] : BigInt(0)
            if r < cum + sub_count
                a[pos] = d
                r -= cum
                cm = new_cm
                remaining -= 1
                found = true
                break
            end
            cum += sub_count
        end
        if !found
            error("Failed to find digit")
        end
    end
    return a
end


# Precompute the number of restricted growth strings for l remaining elements with current max label cm
# dp[l+1, cm+1] stores the value for l elements, current max cm (cm=0 to n-1)
function precompute_dp(n::Int)::Matrix{Int}
    max_label = n - 1
    dp = zeros(Int, n + 1, n + 1)  # Rows: l=0 to n (indices 1 to n+1), Cols: cm=0 to n-1 (indices 1 to n), extra col unused
    dp[1, :] .= 1  # For l=0, all cm
    for l = 1:n
        for cm = 0:n-1
            sum_val = 0
            max_v = min(cm + 1, max_label)
            for v = 0:max_v
                new_cm = max(cm, v)
                sum_val += dp[l, new_cm + 1]  # dp[l, ...] is for l-1
            end
            dp[l + 1, cm + 1] = sum_val
        end
    end
    return dp
end

# Compute the lexicographic index (1-based) of a given restricted growth string a (1-based vector, a[1]=0)
function growth_to_index(a::Vector{Int}, n::Int)::Int
    dp = precompute_dp(n)
    rank = 0
    cm = -1
    for j = 1:n
        max_v = cm + 1
        av = a[j]
        for v = 0:av-1
            new_cm = max(cm, v)
            l_remaining = n - j
            rank += dp[l_remaining + 1, new_cm + 1]
        end
        cm = max(cm, av)
    end
    rank += 1
    bn = bell_number(n)
    rank > bn  && error("Index $rank > Bn($bn) for S($(n)), gs = $(a)")
    return rank
end

# Convert a partition representation (vector of merge trees) to its canonical restricted growth string
function partition_to_growth(p::Vector{Any}, n::Int)::Vector{Int}
    # Get blocks as sorted vectors of elements
    blocks = [flatten_tree(comp) for comp in p]
    # Sort blocks by their minimum element
    sort!(blocks; by = minimum)
    # Assign labels 0 to length(blocks)-1
    label_dict = Dict{Int, Int}()
    for (label, block) in enumerate(blocks)
        for elem in block
            label_dict[elem] = label - 1  # 0-based labels
        end
    end
    # Build the growth string a[1..n]
    a = Vector{Int}(undef, n)
    for i = 1:n
        a[i] = label_dict[i]
    end
    return a
end

# Main function: Maps partition to its lexicographic index (Bell index)
function partition_to_index(p::Vector{Any}, n::Int)::Int
    a = partition_to_growth(p, n)
    return growth_to_index(a, n)
end
