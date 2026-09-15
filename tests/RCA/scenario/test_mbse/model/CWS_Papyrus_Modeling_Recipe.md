# Papyrus Modeling Recipe for CWS (v3.1 KG Alignment)

## Purpose
Step-by-step instructions to build the Circulating Water System (CWS) model in Papyrus aligned with your KG schema.

---

## 1. Project Setup
1. Create Papyrus Project
2. Select UML + SysML
3. Name: CWS_Model
4. Apply SysML profile

---

## 2. Block Definition Diagram (BDD)
Diagram: CWS_Structure_BDD

Create Blocks:
CWS, PumpTrain, ScreenTrain, TrashRemoval, CirculatingWaterPump, PumpMotor,
IntakeValve, DischargeValve, Condenser, MonitoringBoundary,
FlowTransmitter, PressureTransmitter, MotorBreaker, PumpStatusSource

Create Composition:
CWS → PumpTrain (3)
CWS → ScreenTrain (3)
CWS → TrashRemoval (3)
CWS → Condenser (3)
PumpTrain → CirculatingWaterPump
Pump → PumpMotor

---

## 3. Internal Block Diagram (IBD)
Diagram: CWS_Flows_IBD

Add instances:
pumpTrainA/B/C, screenA/B/C, condenserA/B/C, monitor

Add Ports:
Pump: suction, discharge
Valve: inlet, outlet
Condenser: cw_in, cw_out
Motor: power_in
Transmitters: process_in, signal_out
Breaker: power_in/out, status_out
Monitor: signal inputs

Create Flows:
CirculatingWater, MotorPower, FlowSignal, PressureSignal, BreakerStatus, PumpStatus

Connect:
Fluid path → CirculatingWater
Instrumentation → sensing connections
Signals → monitoring boundary

---

## 4. Requirement Diagram
Diagram: CWS_Requirements_RD

Add:
REQ_CWS_001–004

Add satisfy:
CWS → REQ_CWS_001
Pump → REQ_CWS_002
PumpTrain → REQ_CWS_003

---

## 5. Mapping to KG
Block → element_definition
Part → element_usage
Composition → has_part
Port → port
Flow → flow_definition
Connector → connector
Requirement → requirement
Satisfy → satisfies

---

## 6. Rules
- Every connector connects two ports
- Every port has one owner
- Keep diagrams minimal
- Keep documents/specs in KG

---

## 7. Workflow
1. Build BDD
2. Build IBD
3. Add requirements
4. Export to KG
5. Validate

---

## 8. Best Practice
Start with Pump Train A only, then replicate.

---

## Outcome
You get a clean, validated MBSE model aligned with your KG.


