subroutine Merge(A, B, C)
    i = 0  // Index for array A
    j = 0  // Index for array B
    k = 0  // Index for array C

    while i < length(A) and j < length(B):
        if A[i] <= B[j]:
            C[k] = A[i]
            i = i + 1
        else:
            C[k] = B[j]
            j = j + 1
        k = k + 1

    // Collect remaining elements of A, if any
    while i < length(A):
        C[k] = A[i]
        i = i + 1
        k = k + 1

    // Collect remaining elements of B, if any
    while j < length(B):
        C[k] = B[j]
        j = j + 1
        k = k + 1

end subroutine

procedure AllMerge(V, M, Merge, r ∈ C_p^s)
    if p = 1 then
        M ← V return
    end if

    for k = 0, ..., q - 1 do
        ε ← s_{k+1} ∧ 0x1
        t, f ← (r - s_k + ε + p) mod p, (r + s_k - ε) mod p

        if ε = 1 then
            Send(M, t, C_p^s) || Recv(T, f, C_p^s)
            Merge(M, T, M)
        else
            if k = 0 then
                Send(V, t, C_p^s) || Recv(M, f, C_p^s)
            else
                Merge(V, M, M')
                Send(M', t, C_p^s) || Recv(T, f, C_p^s)
                Merge(M, T, M)
            end if
        end if
    end for

    Merge(V, M, M)
end procedure
