**16S High-Current Inductor-Based Active Balancing BMS**



This project features a high-performance Active Battery Management System (BMS) designed for 16S LiFePO4 battery packs.

It utilizes a Power Inductor-based energy transfer mechanism coupled with a cost-optimized Odd/Even Bus Matrix topology to achieve balancing currents up to 5A.



## **Key Features:**



**Topology:** Advanced Odd/Even Switching Matrix (halves the number of MOSFETs compared to traditional full matrices).



**Energy Transfer:** Inductor-based "Buck-Boost" style transfer for high efficiency and high current.



**Power Stage:** MCG052P10Y-TP P-Channel MOSFETs and IAUT300N08S5N011ATMA1 N-Channel MOSFETS for robust performance, thermal and high current stability.



**Polarity Management:** Integrated Bidirectional H-Bridge to correct voltage polarity shifts inherent in Odd/Even bus designs.



**Control Unit:** Powered by an STM32 microcontroller managing nanosecond-precision switching and dead-time insertion.



**Monitoring:** Custom Qt-based GUI for real-time cell voltage visualization, state-of-health (SoH) tracking, and parameter tuning.



**Communication: 2xFDcan,1xUART**



**Component**                  -      **Specification**

Max Cell Count             -      16S (25S versions will be come up with V2.0)

Balancing Current          -      Up to 5A (Adjustable via PWM/Duty Cycle)

MOSFET Rating(balance)     -      "-100V VDS​, 52mΩ RDS(on)​ (MCG052P10Y-TP)"

Inductor                   -      22μH - 47μH (High Saturation Current)

Capacitor                  -      30F 2.7V 2s2p

Safety Features            -      " Over-voltage and Under-voltage protection for each cell and pack, temperature protection , ESD protection for communication ports , short circuit protection"





#### **System Architecture**



**1. Odd/Even Matrix Topology**



The system utilizes a dual-bus architecture where:



Even Nodes (0, 2, 4...) connect to BUS A.



Odd Nodes (1, 3, 5...) connect to BUS B. 



This allows any individual cell to be isolated and connected to the central power inductor using only 2 MOSFET groups per node instead of 4.



**2. H-Bridge \& Polarity Correction**



Since the relative polarity between BUS A and BUS B flips depending on the selected cell (e.g., Cell 1 vs. Cell 2), a central H-Bridge dynamically routes the current to ensure the energy is correctly stored in the inductor and transferred to the target cell or buffer.

