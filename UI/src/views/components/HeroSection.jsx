import landingPageImg from "../../assets/landingPage.png";
import handIcon from "../../assets/klik.png";

export default function HeroSection() {
  return (
    <section className="min-h-screen bg-[#009DFF] text-white pt-0 pb-12 px-6 flex flex-col-reverse md:flex-row items-center justify-between">
      <div className="text-center md:text-left w-full md:w-1/2 ml-0 md:ml-10 mt-6 md:mt-28">
        <h1 className="text-5xl font-bold mb-4">SILENT</h1>
        <p className="text-xl mb-6">
          sign language interpretation <br /> and expression translator
        </p>
        <button
          className="bg-[#007ACC] hover:bg-[#006bb3] px-6 py-3 rounded flex items-center gap-2 transition mx-auto md:mx-0"
          onClick={() => {
            const target = document.getElementById("home");
            if (target) {
              target.scrollIntoView({ behavior: "smooth" });
            }
          }}
        >
          Try me now
          <img src={handIcon} alt="Hand icon" className="w-5 h-6" />
        </button>
      </div>
      <img
        src={landingPageImg}
        alt="Sign illustration"
        className="w-90 mb-8 md:mb-0 mt-30"
      />
    </section>
  );
}
