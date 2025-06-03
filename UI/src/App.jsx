import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from "./views/components/Navbar";
import HeroSection from "./views/components/HeroSection";
import Content from "./views/components/Content";
import Footers from "./views/components/Footers";
import Camera from "./views/components/Camera"

function App() {
  return (
    <Router>
      <Routes>
        <Route
          path="/"
          element={
            <>
              <Navbar />
              <HeroSection />
              <Content />
              <Footers />
            </>
          }
        />
        <Route
          path="/kamera"
          element={
            <>
              <Camera active={true} />
            </>
          }
        />
      </Routes>
    </Router>
  );
}

export default App;
