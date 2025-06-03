export default function createContentPresenter(view) {
  return {
    handleStart(language) {
      view.setSelectedLanguage(language);
      view.setShowTranslate(true);
      setTimeout(() => {
        const el = document.getElementById("translate");
        if (el) el.scrollIntoView({ behavior: "smooth" });
      }, 100);
    },

    handleImageUpload(event) {
      const file = event.target.files[0];
      if (file && file.type.startsWith("image/")) {
        const imageUrl = URL.createObjectURL(file);
        view.setPreviewUrl(imageUrl);
        view.setIsImageReady(true);
      } else {
        alert("Hanya gambar yang diperbolehkan.");
      }
    }
  };
}
