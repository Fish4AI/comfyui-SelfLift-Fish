import { app } from "../../scripts/app.js";

// Display layout only. Keep the widget array and its serialization order intact.
const LABELS = {
    model: "Model", positive: "Positive", negative: "Negative",
    vae: "Video VAE", latent_image: "Target AV latent",
    sampler: "Sampler (Euler)", sigmas: "Sigma schedule",
    model_hires: "Stage 2 model (optional)",
    seed: "Seed", control_after_generate: "Seed behavior",
    cfg: "cfg", transition_step: "Low-resolution steps",
    lowres_scale: "Low-resolution scale", rho: "Pixel correction coverage",
    w_min: "Correction strength min", w_max: "Correction strength max",
    upscaler_model: "Upscaler model", highres_tiling: "Stage 2 memory saving",
    upscaler_unload: "Unload upscaler", highres_tiling_mode: "Memory-saving mode",
    highres_token_chunk: "Token chunk size",
    highres_weight_cache_gb: "Weight cache budget (GiB)",
};
const ADVANCED = new Set([
    "cfg", "rho", "w_min", "w_max", "upscaler_unload",
    "highres_tiling_mode", "highres_token_chunk", "highres_weight_cache_gb",
]);
const MODES = {
    global_stream: "Global stream (recommended)", global: "Global (compatible)",
    spatial: "Spatial (legacy)", global_ram: "RAM staging (legacy)",
};
const SEED_MODES = {
    fixed: "Fixed", increment: "Increment", decrement: "Decrement", randomize: "Randomize",
};
const installed = new WeakSet();
const originalVisibility = new WeakMap();
const TOOLTIPS = {
    transition_step: "Number of low-resolution steps within the upstream total. For example, 6 out of 10 leaves 4 target-resolution steps. More low-resolution steps usually save time but leave fewer steps to refine target-resolution details. Must be smaller than the total step count.",
    lowres_scale: "Low-resolution width and height relative to the target; frame count is unchanged. 0.60 means about 36% of the target spatial positions, subject to rounding, not 60% of total compute. Lower values usually save compute but may lose small details. Tested default: 0.60.",
    rho: "Experimental fraction of high-risk latent locations corrected toward a pixel-VAE anchor, not correction strength. Strength is controlled by the min/max fields. Any positive coverage with positive maximum strength requires VAE decoding and re-encoding of the video; 0.05 does not mean 5% of that cost. Default 0 skips this route. When Upscaler model is None, use positive coverage and positive maximum strength.",
};

function values(node) {
    const get = name => node.widgets?.find(w => w.name === name)?.value;
    return {
        rho: Number(get("rho") ?? 0), max: Number(get("w_max") ?? 1),
        cfg: Number(get("cfg") ?? 1), memory: Boolean(get("highres_tiling")),
        mode: get("highres_tiling_mode"), upscaler: get("upscaler_model"),
    };
}

function refresh(node, resize = true) {
    const nativeAdvanced = typeof node.hasAdvancedWidgets === "function";
    const state = values(node);
    const statuses = [];
    if (state.cfg !== 1) statuses.push(`cfg ${state.cfg}`);
    if (state.rho > 0 && state.max > 0) statuses.push("Pixel correction on");
    else if (state.upscaler === "none") statuses.push("Pixel correction required");
    const toggle = node.widgets?.find(w => w.name === "Advanced settings");
    if (toggle) toggle.label = "Advanced settings" + (!node.showAdvanced && statuses.length ? ` · ${statuses.join(" · ")}` : "");
    for (const w of node.widgets ?? []) {
        if (!LABELS[w.name]) continue;
        w.label = LABELS[w.name];
        if (TOOLTIPS[w.name]) w.options.tooltip = TOOLTIPS[w.name];
        if (w.name === "upscaler_model") w.options.tooltip = state.upscaler === "none"
            ? "None disables the learned latent upscaler. Enable Pixel correction coverage and a positive maximum correction strength in Advanced settings. This route requires VAE decoding and re-encoding. Selecting None expands Advanced settings without changing parameter values."
            : "Learned latent upscaler used at the resolution transition. The tested setup uses h3_upscaler_lms_v0.1.safetensors with Pixel correction coverage 0. Selecting None requires pixel correction and expands Advanced settings.";
        if (!originalVisibility.has(w)) originalVisibility.set(w, {
            type: w.type, computeSize: w.computeSize,
        });
        const advanced = ADVANCED.has(w.name);
        if (nativeAdvanced) w.advanced = advanced;
        // Converted widget inputs belong to ComfyUI; do not change their visibility.
        if (String(w.type).startsWith("converted-widget")) continue;
        const linked = node.inputs?.some(s => s.widget?.name === w.name && s.link != null);
        const correctionInactive = ["w_min", "w_max"].includes(w.name) && state.rho <= 0;
        const memoryInactive = ["highres_tiling_mode", "highres_token_chunk", "highres_weight_cache_gb"].includes(w.name) && !state.memory;
        const cacheInactive = w.name === "highres_weight_cache_gb" && state.mode !== "global_stream";
        // Also hide from legacy-canvas layout, which otherwise reserves blank
        // rows for advanced widgets even when their drawing is suppressed.
        const hidden = !linked && (correctionInactive || memoryInactive || cacheInactive || (advanced && !node.showAdvanced));
        if ("hidden" in w) w.hidden = hidden;
        else {
            const original = originalVisibility.get(w);
            w.type = hidden ? "selflift-hidden" : original.type;
            w.computeSize = hidden ? () => [0, -4] : original.computeSize;
        }
    }
    for (const slot of node.inputs ?? []) {
        if (LABELS[slot.name]) slot.label = LABELS[slot.name];
    }
    for (const slot of node.outputs ?? []) slot.label = "AV latent";
    if (resize && node.graph && node.computeSize && node.setSize) {
        const minimum = node.computeSize();
        node.setSize([Math.max(node.size?.[0] ?? 0, minimum[0], 380), minimum[1]]);
    }
    node.setDirtyCanvas?.(true, true);
}

function install(node) {
    if (installed.has(node)) return;
    installed.add(node);
    node.showAdvanced = false;
    // This extra UI button has no backend parameter and no serialized value.
    const toggle = node.addWidget("button", "Advanced settings", null, () => {
        node.showAdvanced = !node.showAdvanced;
        refresh(node);
    }, { serialize: false });
    toggle.serialize = false;
    toggle.options.tooltip = "Expand or collapse advanced controls below this button. Collapsing changes visibility only; all values remain active and are saved with the workflow.";
    // The layout hook controls row positions without changing node.widgets,
    // positional workflow values, widget names, or backend parameter mapping.
    const layoutWidgets = node.getLayoutWidgets;
    if (layoutWidgets) node.getLayoutWidgets = function (...args) {
        const widgets = layoutWidgets.apply(this, args);
        return [
            ...widgets.filter(w => w !== toggle && !ADVANCED.has(w.name)),
            ...widgets.filter(w => w === toggle),
            ...widgets.filter(w => ADVANCED.has(w.name)),
        ];
    };
    const nativeToggle = node.toggleAdvanced;
    if (nativeToggle) node.toggleAdvanced = function (...args) {
        const result = nativeToggle.apply(this, args);
        refresh(this);
        return result;
    };
    for (const name of ["rho", "w_max", "cfg", "highres_tiling", "highres_tiling_mode", "upscaler_model"]) {
        const widget = node.widgets?.find(w => w.name === name);
        if (!widget) continue;
        const callback = widget.callback;
        widget.callback = function (...args) {
            const result = callback?.apply(this, args);
            if (name === "upscaler_model" && widget.value === "none") node.showAdvanced = true;
            refresh(node);
            return result;
        };
    }
    for (const w of node.widgets ?? []) {
        if (w.name === "highres_tiling_mode") w.options.getOptionLabel = value => MODES[value] ?? value;
        if (w.name === "control_after_generate") {
            w.options.getOptionLabel = value => SEED_MODES[value] ?? value;
            w.options.tooltip = "Fixed reuses the seed. Randomize chooses a new seed after each run. Increment/Decrement changes it sequentially. Use Fixed for comparisons.";
        }
        if (w.name === "upscaler_model") w.options.getOptionLabel = value => value === "none" ? "None (requires pixel correction)" : value;
    }
    // Modern frontend restores widgets by name or position after node creation.
    // Refresh only labels/visibility after restoration, never write defaults here.
    const configure = node.onConfigure;
    node.onConfigure = function (...args) {
        const result = configure?.apply(this, args);
        // Expansion is presentation state; keep all restored parameter values.
        this.showAdvanced = false;
        if (["SelfLift Progressive Sampler (MiniMax H3)", "SelfLift 渐进采样（H3）"].includes(this.title)) this.title = "selflift-Fish";
        refresh(this, false);
        queueMicrotask(() => refresh(this));
        return result;
    };
    refresh(node, false);
    queueMicrotask(() => refresh(node));
}

app.registerExtension({
    name: "SelfLift.Fish.CompactUI",
    nodeCreated(node) {
        if ((node.comfyClass || node.type) === "SelfLiftH3Sampler") install(node);
    },
});
