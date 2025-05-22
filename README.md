# A Digital Twin for Personalized Type-1 Diabetes Care

This project implements a **Digital Twin** system for optimizing insulin delivery in **Type-1 Diabetes (T1D)** patients. It leverages an **LSTM-based PID tuning model** trained on real glucose data and simulates a closed-loop insulin pump. The system supports real-time testing using predefined patient scenarios and can be deployed on embedded hardware (Jetson TX2) for Hardware-in-the-Loop (HIL) testing.

---

## 🚀 Project Components

### 1. **PID Tuning Model**
- Predicts PID gains (Kp, Ki, Kd) using LSTM
- Trained on patient glucose and time data
- Technologies: `TensorFlow`, `NumPy`, `Pandas`

### 2. **Insulin Delivery Simulator**
- Simulates basal and bolus insulin administration
- Uses predictions from the PID model
- Controlled via patient meal/test case inputs

### 3. **Test Case Manager**
- Inputs predefined patient weight and meal schedules
- Format: plain `.txt` file (`TestCases.txt`)
- Drives simulation in a reproducible way

---

## 🧠 Key Tools and Libraries

| Purpose                  | Tools & Technologies                          |
|--------------------------|-----------------------------------------------|
| Deep Learning            | TensorFlow, LSTM                              |
| Feature Engineering      | Cyclic Encoding (sin/cos)                     |
| Simulation Modeling      | Simcenter Amesim, FMI                         |
| Real-Time Control Logic  | Python                                        |
| Hardware Deployment      | VSI, NVIDIA Jetson TX2                        |
| Data Format              | XML (Glucose Data), TXT (TestCases)          |

---

## 📄 How to Run

### 1. Install Requirements
```bash
pip install -r requirements.txt
```
### 2. Train the PID Model (Optional)
Run: `notebooks/PID_Tuning_for_Artificial_Pancreas_with_LSTM.ipynb`

Or use the pretrained model in `models/`
### 3. Run Insulin Simulation
```bash
python src/artificial_pancreas_simulator.py
```
