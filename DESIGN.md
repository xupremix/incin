# Dyn Parameters
Dyn indicates any runtime known parameter
- Shape
- RequiresGrad
- DType
- Device

of course operations with dyn cannot have compile time guarantees and methods will be wrapped in results
(there is always the problem of running out of memory for allocations when performing operations)

## Shape
Generic: Dyn
Field: Dyn
take as param: anything list-like

## RequiresGrad
Generic: Dyn
Field: bool
take as param: true or false

## DType
Generic: Dyn
Field: Dyn
take as param: anything which can be intepreted ad a DType
( DType enum )

## Device
Generic: Dyn
Field: Dyn
take as param: anything which can be intepreted ad a device
( Device enum )

# Not Dyn Parameters

## Shape

### Partially Dynamic
Generic: (Const<N>, usize, ...) where usize denotes a dynamic parameter
Field: (Const<N>, usize, ...)
take as param: the Field

### Static 
Generic: (Const<N>, Const<M>, ...) only
Field: Nothing
take as param: Nothing

## RequiresGrad
Generic: Grad / NoGrad
Field: PhantomData
take as param: Nothing

## DType
Generic: f32 / u32 ...
Field: PhantomData
take as param: Nothing

## Device
Generic: Cpu / Cuda<N> ...
Field: PhantomData
take as param: Nothing

# Tensor
Generic over
- Shape
- DType
- Device
- RequiresGrad

for a fully dynamic tensor
Tensor<Dyn, Dyn, Dyn, Dyn>

-------------

Shapes


new() -> Self // no parameters just for const shapes
