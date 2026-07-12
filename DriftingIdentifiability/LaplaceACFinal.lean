import DriftingIdentifiability.LaplaceACPropagation

/-!
# Final assembly for the absolutely-continuous Laplace converse

This file contains the final, axiom-free socket for the a.c. Laplace route
documented in `LaplaceACDerivation.md`.

The upstream files have already proved the hard analytic packages:

* L6 kills the two outer rays of the Wronskian;
* L8 kills the two flanks of each upward crossing;
* L9 turns a finite alternating down/up cover into global Wronskian vanishing;
* the certified Wronskian gate turns global Wronskian vanishing into `p = q`.

The theorem here intentionally does not hide the remaining modelling input:
callers must provide the finite alternating cover for the actual Wronskian.
That cover is exactly the output expected from applying the L6 and L8
certificates to the chosen finite breakpoint list.
-/

open MeasureTheory Set Filter Topology

namespace DriftingIdentifiability

open Paper

/-- Final deterministic certificate for the a.c. Laplace Wronskian endgame.

`breaks` is the finite alternating sign-change list, intended to have shape
`[down₁, up₂, down₃, ..., down_M]`.  The `alternating_cover` field says that
the actual normalizer Wronskian vanishes on the left ray, on both flanks of
each upward crossing, and on the right ray.  L9 then fills in the breakpoint
values by continuity. -/
structure LaplaceACFinalAssembly (τ : ℝ) (p q : Measure ℝ) where
  breaks : List ℝ
  wronskian_continuous :
    Continuous (fun x : ℝ => laplaceKernelNormalizerWronskian τ p q x)
  alternating_cover :
    VanishesOnAlternatingUpwardPairs
      (fun x : ℝ => laplaceKernelNormalizerWronskian τ p q x) breaks

/-- The final assembly certificate forces the actual normalizer Wronskian to
vanish everywhere. -/
theorem laplaceAC_wronskian_eq_zero_of_finalAssembly
    (τ : ℝ) (p q : Measure ℝ)
    (h : LaplaceACFinalAssembly τ p q) :
    ∀ x : ℝ, laplaceKernelNormalizerWronskian τ p q x = 0 := by
  exact continuous_eq_zero_of_alternatingUpwardPairs h.breaks
    h.wronskian_continuous h.alternating_cover

/-- **Final a.c. Laplace assembly gate.**  Once the L6/L8/L9 deterministic
certificate has been built for the actual Wronskian, the certified Wronskian
injectivity theorem gives `p = q`.

Zero drift is not repeated here because it is used upstream to produce the
assembly certificate; after that certificate is available, the Wronskian gate
itself needs only probability measures and a valid bandwidth. -/
theorem laplaceAC_identifies_of_finalAssembly
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (h : LaplaceACFinalAssembly τ p q) :
    p = q :=
  laplaceKernelNormalizer_wronskian_eq_zero_imp_eq τ hτ p q
    (laplaceAC_wronskian_eq_zero_of_finalAssembly τ p q h)

/-- Convenience form for the single-crossing/3A case.  If one downward
breakpoint has zero Wronskian on both outer rays, it gives the alternating cover
expected by the final assembly theorem. -/
theorem VanishesOnAlternatingUpwardPairs.singleDown
    {W : ℝ → ℝ} {a : ℝ}
    (hleft : ∀ x : ℝ, x < a → W x = 0)
    (hright : ∀ x : ℝ, a < x → W x = 0) :
    VanishesOnAlternatingUpwardPairs W [a] := by
  exact ⟨hleft, hright⟩

/-- Convenience form for the three-crossing/first nontrivial 3B case.  A
down/up/down list is covered by the left outer ray, the two flanks of the upward
crossing, and the right outer ray. -/
theorem VanishesOnAlternatingUpwardPairs.downUpDown
    {W : ℝ → ℝ} {a b c : ℝ}
    (hleft : ∀ x : ℝ, x < a → W x = 0)
    (hleftFlank : ∀ x : ℝ, a < x → x < b → W x = 0)
    (hrightFlank : ∀ x : ℝ, b < x → x < c → W x = 0)
    (hright : ∀ x : ℝ, c < x → W x = 0) :
    VanishesOnAlternatingUpwardPairs W [a, b, c] := by
  exact ⟨hleft, hleftFlank, hrightFlank, hright⟩

end DriftingIdentifiability
