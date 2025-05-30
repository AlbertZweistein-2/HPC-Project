subroutine Merge(A, B, C, (compare))
    //suppose length(A) = length(B) => length(C) = 2*length(A)

    for i in length(C) do
        if compare(A[i],B[i]) <= 0 then // A[i] <= B[i]
            C[i] = A[i]
end subroutine

procedure AllReduceMerge(V, W, ⊕, r ∈ C_p^s)
    if p = 1 then
        W ← V return
    end if

    for k = 0, ..., q - 1 do
        ε ← s_{k+1} ∧ 0x1
        t, f ← (r - s_k + ε + p) mod p, (r + s_k - ε) mod p

        if ε = 1 then
            Send(V, t, C_p^s) || Recv(T, f, C_p^s)
            Merge(W, T, W)
        else
            if k = 0 then
                Send(V, t, C_p^s) || Recv(W, f, C_p^s)
            else
                W' ← Merge(V, W)
                Send(W', t, C_p^s) || Recv(T, f, C_p^s)
                Merge(W, T, W)
            end if
        end if
    end for

    Merge(V, W, W)
end procedure
