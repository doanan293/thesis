// Browser tests render with the app stylesheet, so layout, stacking and visibility match the app
// (for example a dialog popup sits above Base UI's modal backdrop, as it does for users). The CSS
// is inlined so it applies before any test renders.
import appStyles from "../../app/app.css?inline"

const style = document.createElement("style")
style.dataset["testStylesheet"] = "app"
style.textContent = appStyles
document.head.append(style)
