## What we have said
* Reynolds not too high/ middle

# TODO
* VMS
* FIlter instead of identity sigma_star
* comparison computational cost vs error (giacomo's code vs lps)
* Reduction

## Problems
* Projection on test function is not possible

Idea: transfer the projection onto the trial function. Using the dual operator.
I want to compute $(z, \Pi w)$ where $w$ depends on the test functions but $z$ not.
This can be written as 
$(z,\Pi w ) = z^T S \Pi w $
so, we are looking for an $\hat{\Pi}$ such that
$(z,\Pi w ) = z^T S \Pi w =(\hat{\Pi}z, w) = z^T \hat{\Pi}^T S w.$

Hence, we have to solve $\hat{\Pi}:=S^{-T}\Pi^T S^{T}.$ Likely, $S$ is symmetric and it is the matrix of the scalar product related to the projected values (might be also related to two different spaces, hence, a rectangular matrix **carefull**). If $\Pi$ is a projection from a high dimensional space to a low one, and its tranposition should be the embedding from the low dimensional space to the high order one. This should give too many troubles, we just have to be carefull with the spaces.
My biggest question is how to deal with the matrices in FEniCS, without building them, but writing them as bilinear forms and so on. 
