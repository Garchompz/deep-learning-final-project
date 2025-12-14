import { useState } from 'react'
import DotGrid from './background'
import './index.css'

function App() {
  const [image, setImage] = useState(null);
  const [caption, setCaption] = useState('');
  const [loading, setLoading] = useState(false);

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setImage(e.target.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const generateCaption = async () => {
    if (!image) return;
    
    setLoading(true);
    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: image
        })
      });

      const data = await response.json();
      if (data.error) {
        setCaption('Error: ' + data.error);
      } else {
        setCaption(data.caption || 'No caption generated');
      }
    } catch (error) {
      setCaption('Error generating caption. Please try again.');
      console.error('Error:', error);
    }
    setLoading(false);
  };

  const copyCaption = () => {
    if (!caption) return;
    try {
      navigator.clipboard.writeText(caption);
    } catch (e) {
      console.error('Clipboard write failed', e);
    }
  };

  const regenerate = () => {
    generateCaption();
  };

  return (
    <div className="app">
      <DotGrid
        dotSize={1}
        gap={15}
        baseColor="#ffffff69"
        activeColor="#3d3d3dff"
        proximity={30}
        shockRadius={20}
        shockStrength={5}
        resistance={750}
        returnDuration={1.5}
        style={{ position: 'fixed', top: 0, left: 0, width: '100%', height: '100%', zIndex: -1 }}
      />

      <h1>
        AI Image Captioning
      </h1>

      <main className="main-content">
        <div className="content-wrapper">
          <div className="upload-section">
            <h2>Upload Image Here</h2>
            {image ? (
              <div className="image-preview">
                <img src={image} alt="Uploaded" />
              </div>
            ) : (
              <div className="image-placeholder">
                <span>📸</span>
              </div>
            )}
            <input
              type="file"
              id="file-input"
              accept="image/*"
              onChange={handleImageUpload}
              style={{ display: 'none' }}
            />
            <button
              className="choose-btn"
              onClick={() => document.getElementById('file-input').click()}
            >
              Choose File
            </button>
          </div>

          <div className="caption-section">
            <h2>Generated Caption</h2>
            <div className="caption-box">
              {loading ? 'Generating caption...' : caption || 'Your generated caption will appear here'}
            </div>
            <div className="button-group">
              <button className="copy-btn" onClick={copyCaption} disabled={!caption}>
                Copy Caption
              </button>
              <button className="regenerate-btn" onClick={regenerate} disabled={!caption}>
                Regenerate
              </button>
            </div>
          </div>
        </div>

        <button
          className="generate-btn"
          onClick={generateCaption}
          disabled={!image || loading}
        >
          {loading ? 'Generating...' : 'Generate Caption'}
        </button>
      </main>
    </div>
  )
}

export default App
