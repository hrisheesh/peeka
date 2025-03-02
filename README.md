# Peeka - AI-Powered Avatar Application

Peeka is an innovative AI-powered avatar application that creates personalized digital avatars for video conferencing platforms. Using advanced facial feature detection and deep learning models, Peeka generates realistic avatars that can accurately represent users in virtual meetings.

## Key Features

- **Personalized Avatar Creation**: Generate custom avatars from a short 15-30 second training video
- **Facial Feature Detection**: Accurate tracking of facial landmarks and expressions
- **Real-time Lip Synchronization**: Seamless lip movement synchronization with audio input
- **Expression Mapping**: Natural replication of user's facial expressions
- **Head Pose Estimation**: Accurate tracking of head movements and orientation

## System Requirements

### Hardware Requirements
- CPU: Multi-core processor (Intel i5/i7 or AMD equivalent)
- RAM: 16GB recommended (8GB minimum)
- GPU: NVIDIA GPU with CUDA support for optimal performance
- Webcam: HD camera for facial feature capture

### Software Requirements
- Python 3.8 or higher
- CUDA Toolkit (for GPU acceleration)
- dlib and associated dependencies
- Virtual camera drivers (OBS Virtual Camera or similar)

## Project Structure

```
peeka/
├── src/
│   ├── capture/            # Video capture and facial detection
│   ├── preprocessing/      # Feature extraction pipeline
│   ├── models/            # Avatar generation models
│   ├── inference/         # Real-time processing engine
│   ├── virtual_camera/    # Camera output integration
│   └── ui/               # Training and control interface
├── docs/                 # Technical documentation
└── requirements.txt      # Python dependencies
```

## Quick Start

1. **Setup Environment**
```bash
# Clone repository
git clone https://github.com/hrisheesh/peeka
cd peeka

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> **Note**: Model files are not included in the repository due to their size. They will be generated automatically during the training process.

2. **Training Your Avatar**
- Launch the training interface: `python src/main.py`
- Follow the on-screen instructions to record your training video
- Wait for the avatar generation process to complete

3. **Using Your Avatar**
- Start the Peeka application
- Select your generated avatar
- Choose your target video conferencing application
- Begin your virtual meeting with your personalized avatar

## Development Status

Peeka is currently in active development with the following milestones:

- [x] Core facial feature detection implementation
- [x] Basic training interface
- [x] Facial feature model training pipeline
- [x] Training visualization and results panel
- [ ] Avatar generation model integration
- [ ] Real-time lip sync implementation
- [ ] Virtual camera integration
- [ ] Expression synthesis refinement

## Contributing

We welcome contributions! If you'd like to help improve Peeka:

1. Fork the repository
2. Create your feature branch
3. Implement your changes
4. Submit a pull request

Please refer to our contribution guidelines in the docs folder for more details.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For bug reports and feature requests, please use the GitHub issues page.