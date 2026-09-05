# Canonical operation coverage

This file is generated from `incin_core::exec::OPERATION_CATALOG`; the Rust catalog is authoritative.

- Canonical operations: 174
- Backend-executable operations: 164
- Non-backend execution sites: 10

| Execution site | Operations |
|---|---:|
| `Kernel` | 146 |
| `Creation` | 13 |
| `HostReadback` | 5 |
| `Composed` | 3 |
| `Mutation` | 3 |
| `DeviceTransfer` | 1 |
| `GraphState` | 3 |

## Non-backend operations

| Operation | Site | Reason |
|---|---|---|
| `sample` | `Composed` | the frontend composition owns the execution semantics |
| `to_device` | `DeviceTransfer` | produces storage on another backend, which the executor cannot name |
| `require_grad` | `GraphState` | acts on autograd state, not on an allocation |
| `detach` | `GraphState` | acts on autograd state, not on an allocation |
| `backward` | `GraphState` | acts on autograd state, not on an allocation |
| `rnn` | `Composed` | the frontend composition owns the execution semantics |
| `lstm` | `Composed` | the frontend composition owns the execution semantics |
| `sgd_step` | `Mutation` | writes through an operand; execution borrows operands shared |
| `adam_step` | `Mutation` | writes through an operand; execution borrows operands shared |
| `adamw_step` | `Mutation` | writes through an operand; execution borrows operands shared |
