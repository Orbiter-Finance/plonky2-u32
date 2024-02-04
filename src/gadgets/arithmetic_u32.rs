use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use plonky2::util::ceil_div_usize;
use core::marker::PhantomData;
use plonky2::plonk::circuit_data::CommonCircuitData;
use plonky2::util::serialization::{Buffer, IoResult, Read, Write};

use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::generator::{GeneratedValues, SimpleGenerator};
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::{PartitionWitness, Witness};
use plonky2::plonk::circuit_builder::CircuitBuilder;

use crate::gates::add_many_u32::U32AddManyGate;
use crate::gates::arithmetic_u32::U32ArithmeticGate;
use crate::gates::subtraction_u32::U32SubtractionGate;
use crate::gates::comparison::ComparisonGate;
use crate::serialization::{ReadU32, WriteU32};
use crate::witness::GeneratedValuesU32;

#[derive(Clone, Copy, Debug, Default)]
pub struct U32Target(pub Target);

pub trait CircuitBuilderU32<F: RichField + Extendable<D>, const D: usize> {
    fn add_virtual_u32_target(&mut self) -> U32Target;

    fn add_virtual_u32_targets(&mut self, n: usize) -> Vec<U32Target>;

    /// Returns a U32Target for the value `c`, which is assumed to be at most 32 bits.
    fn constant_u32(&mut self, c: u32) -> U32Target;

    fn zero_u32(&mut self) -> U32Target;

    fn one_u32(&mut self) -> U32Target;

    fn connect_u32(&mut self, x: U32Target, y: U32Target);

    fn assert_zero_u32(&mut self, x: U32Target);

    /// Checks for special cases where the value of
    /// `x * y + z`
    /// can be determined without adding a `U32ArithmeticGate`.
    fn arithmetic_u32_special_cases(
        &mut self,
        x: U32Target,
        y: U32Target,
        z: U32Target,
    ) -> Option<(U32Target, U32Target)>;

    // Returns x * y + z.
    fn mul_add_u32(&mut self, x: U32Target, y: U32Target, z: U32Target) -> (U32Target, U32Target);

    fn add_u32(&mut self, a: U32Target, b: U32Target) -> (U32Target, U32Target);

    fn add_many_u32(&mut self, to_add: &[U32Target]) -> (U32Target, U32Target);

    fn add_u32s_with_carry(
        &mut self,
        to_add: &[U32Target],
        carry: U32Target,
    ) -> (U32Target, U32Target);

    fn mul_u32(&mut self, a: U32Target, b: U32Target) -> (U32Target, U32Target);

    // Returns x - y - borrow, as a pair (result, borrow), where borrow is 0 or 1 depending on whether borrowing from the next digit is required (iff y + borrow > x).
    fn sub_u32(&mut self, x: U32Target, y: U32Target, borrow: U32Target) -> (U32Target, U32Target);

    fn is_less_than_u32(&mut self, a: U32Target, b: U32Target, num_bits: usize) -> BoolTarget;

    fn check_less_than_u32(&mut self, a: U32Target, b: U32Target,  num_bits: usize);

    fn select_u32(&mut self, b: BoolTarget, x: U32Target, y: U32Target) -> U32Target;

    fn check_less_than_safe_u32(&mut self, a: U32Target, b: u32);

    fn copy_many_u32(&mut self, data: &[U32Target]) -> Vec<U32Target>;

}

impl<F: RichField + Extendable<D>, const D: usize> CircuitBuilderU32<F, D>
    for CircuitBuilder<F, D>
{
    fn add_virtual_u32_target(&mut self) -> U32Target {
        U32Target(self.add_virtual_target())
    }

    fn add_virtual_u32_targets(&mut self, n: usize) -> Vec<U32Target> {
        self.add_virtual_targets(n)
            .into_iter()
            .map(U32Target)
            .collect()
    }

    /// Returns a U32Target for the value `c`, which is assumed to be at most 32 bits.
    fn constant_u32(&mut self, c: u32) -> U32Target {
        U32Target(self.constant(F::from_canonical_u32(c)))
    }

    fn zero_u32(&mut self) -> U32Target {
        U32Target(self.zero())
    }

    fn one_u32(&mut self) -> U32Target {
        U32Target(self.one())
    }

    fn connect_u32(&mut self, x: U32Target, y: U32Target) {
        self.connect(x.0, y.0)
    }

    fn assert_zero_u32(&mut self, x: U32Target) {
        self.assert_zero(x.0)
    }

    /// Checks for special cases where the value of
    /// `x * y + z`
    /// can be determined without adding a `U32ArithmeticGate`.
    fn arithmetic_u32_special_cases(
        &mut self,
        x: U32Target,
        y: U32Target,
        z: U32Target,
    ) -> Option<(U32Target, U32Target)> {
        let x_const = self.target_as_constant(x.0);
        let y_const = self.target_as_constant(y.0);
        let z_const = self.target_as_constant(z.0);

        // If both terms are constant, return their (constant) sum.
        let first_term_const = if let (Some(xx), Some(yy)) = (x_const, y_const) {
            Some(xx * yy)
        } else {
            None
        };

        if let (Some(a), Some(b)) = (first_term_const, z_const) {
            let sum = (a + b).to_canonical_u64();
            let (low, high) = (sum as u32, (sum >> 32) as u32);
            return Some((self.constant_u32(low), self.constant_u32(high)));
        }

        None
    }

    // Returns x * y + z.
    fn mul_add_u32(&mut self, x: U32Target, y: U32Target, z: U32Target) -> (U32Target, U32Target) {
        if let Some(result) = self.arithmetic_u32_special_cases(x, y, z) {
            return result;
        }

        let gate = U32ArithmeticGate::<F, D>::new_from_config(&self.config);
        let (row, copy) = self.find_slot(gate, &[], &[]);

        self.connect(Target::wire(row, gate.wire_ith_multiplicand_0(copy)), x.0);
        self.connect(Target::wire(row, gate.wire_ith_multiplicand_1(copy)), y.0);
        self.connect(Target::wire(row, gate.wire_ith_addend(copy)), z.0);

        let output_low = U32Target(Target::wire(row, gate.wire_ith_output_low_half(copy)));
        let output_high = U32Target(Target::wire(row, gate.wire_ith_output_high_half(copy)));

        (output_low, output_high)
    }

    fn add_u32(&mut self, a: U32Target, b: U32Target) -> (U32Target, U32Target) {
        let one = self.one_u32();
        self.mul_add_u32(a, one, b)
    }

    fn copy_many_u32(&mut self, data: &[U32Target]) -> Vec<U32Target> {
        let mut result = Vec::new();
        for x in data {
            let copy = self.add_virtual_u32_target();
            self.connect_u32(copy, *x);
            result.push(copy);
        }
        result
    }

    fn add_many_u32(&mut self, to_add: &[U32Target]) -> (U32Target, U32Target) {
        match to_add.len() {
            0 => (self.zero_u32(), self.zero_u32()),
            1 => (to_add[0], self.zero_u32()),
            2 => self.add_u32(to_add[0], to_add[1]),
            _ => {
                let num_addends = to_add.len();
                let gate = U32AddManyGate::<F, D>::new_from_config(&self.config, num_addends);
                let (row, copy) =
                    self.find_slot(gate, &[F::from_canonical_usize(num_addends)], &[]);

                for j in 0..num_addends {
                    self.connect(
                        Target::wire(row, gate.wire_ith_op_jth_addend(copy, j)),
                        to_add[j].0,
                    );
                }
                let zero = self.zero();
                self.connect(Target::wire(row, gate.wire_ith_carry(copy)), zero);

                let output_low = U32Target(Target::wire(row, gate.wire_ith_output_result(copy)));
                let output_high = U32Target(Target::wire(row, gate.wire_ith_output_carry(copy)));

                (output_low, output_high)
            }
        }
    }

    fn add_u32s_with_carry(
        &mut self,
        to_add: &[U32Target],
        carry: U32Target,
    ) -> (U32Target, U32Target) {
        if to_add.len() == 1 {
            return self.add_u32(to_add[0], carry);
        }

        let num_addends = to_add.len();

        let gate = U32AddManyGate::<F, D>::new_from_config(&self.config, num_addends);
        let (row, copy) = self.find_slot(gate, &[F::from_canonical_usize(num_addends)], &[]);

        for j in 0..num_addends {
            self.connect(
                Target::wire(row, gate.wire_ith_op_jth_addend(copy, j)),
                to_add[j].0,
            );
        }
        self.connect(Target::wire(row, gate.wire_ith_carry(copy)), carry.0);

        let output = U32Target(Target::wire(row, gate.wire_ith_output_result(copy)));
        let output_carry = U32Target(Target::wire(row, gate.wire_ith_output_carry(copy)));

        (output, output_carry)
    }

    fn mul_u32(&mut self, a: U32Target, b: U32Target) -> (U32Target, U32Target) {
        let zero = self.zero_u32();
        self.mul_add_u32(a, b, zero)
    }

    // Returns x - y - borrow, as a pair (result, borrow), where borrow is 0 or 1 depending on whether borrowing from the next digit is required (iff y + borrow > x).
    fn sub_u32(&mut self, x: U32Target, y: U32Target, borrow: U32Target) -> (U32Target, U32Target) {
        let gate = U32SubtractionGate::<F, D>::new_from_config(&self.config);
        let (row, copy) = self.find_slot(gate, &[], &[]);

        self.connect(Target::wire(row, gate.wire_ith_input_x(copy)), x.0);
        self.connect(Target::wire(row, gate.wire_ith_input_y(copy)), y.0);
        self.connect(
            Target::wire(row, gate.wire_ith_input_borrow(copy)),
            borrow.0,
        );

        let output_result = U32Target(Target::wire(row, gate.wire_ith_output_result(copy)));
        let output_borrow = U32Target(Target::wire(row, gate.wire_ith_output_borrow(copy)));

        (output_result, output_borrow)
    }

    /// Constratins whether `a` is in `[0, b)`, and returns 1 if `a` < `b`, otherwise 0
    /// The ComparisionGate provides function to check x <= y, so if we want to check that a < b equivalent !(b <=a ) 
    fn is_less_than_u32(&mut self, a: U32Target, b: U32Target, num_bits: usize) -> BoolTarget{

        let chunk_bits = 2;
        let num_chunks = ceil_div_usize(num_bits, chunk_bits);

        let b_le_a_gate = ComparisonGate::new(num_bits, num_chunks);
        let b_le_a_row = self.add_gate(b_le_a_gate.clone(), vec![]);
        self.connect(
            Target::wire(b_le_a_row, b_le_a_gate.wire_first_input()),
            b.0,
        );
        self.connect(
            Target::wire(b_le_a_row, b_le_a_gate.wire_second_input()),
            a.0,
        );
        let b_le_a_result = Target::wire(b_le_a_row, b_le_a_gate.wire_result_bool());

        let result = self.not(BoolTarget::new_unsafe(b_le_a_result));
        result
    }

    fn check_less_than_safe_u32(&mut self, a: U32Target, b: u32) {
        let b_target = self.constant_u32(b);
        let num_bits = 32;
        self.range_check(a.0, num_bits);
        self.check_less_than_u32(a, b_target, num_bits)
    }

    /// Constraints that `a` < `b`, same as is_less_than_u32 but asserts that !(b <= a)
    fn check_less_than_u32(&mut self, a: U32Target, b: U32Target,  num_bits: usize) {

        let chunk_bits = 2;
        let num_chunks = ceil_div_usize(num_bits, chunk_bits);

        let b_le_a_gate = ComparisonGate::new(num_bits, num_chunks);
        let b_le_a_row = self.add_gate(b_le_a_gate.clone(), vec![]);
        self.connect(
            Target::wire(b_le_a_row, b_le_a_gate.wire_first_input()),
            b.0,
        );
        self.connect(
            Target::wire(b_le_a_row, b_le_a_gate.wire_second_input()),
            a.0,
        );
        let b_le_a_result = Target::wire(b_le_a_row, b_le_a_gate.wire_result_bool());

       self.assert_zero(b_le_a_result)
    }

    fn select_u32(&mut self, b: BoolTarget, x: U32Target, y: U32Target) -> U32Target {
        let result_target = self.select(b, x.0, y.0);
        U32Target(result_target)
    }
}

#[derive(Debug)]
struct SplitToU32Generator<F: RichField + Extendable<D>, const D: usize> {
    x: Target,
    low: U32Target,
    high: U32Target,
    _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D>
    for SplitToU32Generator<F, D>
{
    fn id(&self) -> String {
        "SplitToU32Generator".to_string()
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_target(self.x)?;
        dst.write_target_u32(self.low)?;
        dst.write_target_u32(self.high)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let x = src.read_target()?;
        let low = src.read_target_u32()?;
        let high = src.read_target_u32()?;
        Ok(Self {
            x,
            low,
            high,
            _phantom: PhantomData,
        })
    }

    fn dependencies(&self) -> Vec<Target> {
        vec![self.x]
    }

    fn run_once(&self, witness: &PartitionWitness<F>, out_buffer: &mut GeneratedValues<F>) {
        let x = witness.get_target(self.x);
        let x_u64 = x.to_canonical_u64();
        let low = x_u64 as u32;
        let high = (x_u64 >> 32) as u32;

        out_buffer.set_u32_target(self.low, low);
        out_buffer.set_u32_target(self.high, high);
    }
}

#[cfg(test)]
mod tests {

    use std::{dbg, println};

    use anyhow::{Ok, Result};
    use plonky2::iop::witness::PartialWitness;
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    use rand::rngs::OsRng;
    use rand::Rng;
    use super::*;

    #[test]
    pub fn test_check_less_than_u32() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let config = CircuitConfig::standard_recursion_config();

        let pw = PartialWitness::new();
        let mut builder = CircuitBuilder::<F, D>::new(config);

        let a: u32 = 1_u32;
        let b: u32 = 2_u32;

        let a_target = builder.constant_u32(a);
        let b_target = builder.constant_u32(b);

        builder.check_less_than_u32(a_target, b_target, 32);

        let data = builder.build::<C>();
        let proof = data.prove(pw).unwrap();
        data.verify(proof).unwrap();
        
        Ok(())
    }

    #[test]
    fn test_is_less_than_u32() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let config = CircuitConfig::standard_recursion_config();

        let pw = PartialWitness::new();
        let mut builder = CircuitBuilder::<F, D>::new(config);

        let a: u32 = 1_u32;
        let b: u32 = 2_u32;

        let a_target = builder.constant_u32(a);
        let b_target = builder.constant_u32(b);
        let expect_result = builder.constant_bool(a < b);

        // let expect_result_2 = builder.constant_bool(a < b);

        let result = builder.is_less_than_u32(a_target, b_target, 32);
        builder.connect_u32(U32Target(result.target), U32Target(expect_result.target));

        let data = builder.build::<C>();
        let proof = data.prove(pw).unwrap();
        data.verify(proof).unwrap();
        Ok(())
    }

    #[test]
    pub fn test_add_many_u32s() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        const NUM_ADDENDS: usize = 15;

        let config = CircuitConfig::standard_recursion_config();

        let pw = PartialWitness::new();
        let mut builder = CircuitBuilder::<F, D>::new(config);

        let mut rng = OsRng;
        let mut to_add = Vec::new();
        let mut sum = 0u64;
        for _ in 0..NUM_ADDENDS {
            let x: u32 = rng.gen();
            sum += x as u64;
            to_add.push(builder.constant_u32(x));
        }
        let carry = builder.zero_u32();
        let (result_low, result_high) = builder.add_u32s_with_carry(&to_add, carry);
        let expected_low = builder.constant_u32((sum % (1 << 32)) as u32);
        let expected_high = builder.constant_u32((sum >> 32) as u32);

        builder.connect_u32(result_low, expected_low);
        builder.connect_u32(result_high, expected_high);

        let data = builder.build::<C>();
        let proof = data.prove(pw).unwrap();
        data.verify(proof)
    }
}
