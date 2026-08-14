/**
 * Settings for the 🎥 Animate action on gallery cards (POST /api/animate).
 * Nothing here submits — the card does, reading these values from the store,
 * exactly like `RefineBar`.
 *
 * Two engines, and each hides the knobs the other owns:
 *
 * · **AnimateDiff** — a motion module on the SAME SD1.5 UNet everything else
 *   uses, so an image can be animated from its own promptless conditioning with
 *   no text encoder in the loop. ~20s, 2s of video, keeps the source's own
 *   resolution (a flux upscale stays 768). sd15/sd2 sources only. Motion is
 *   drift and atmosphere: the composition holds still.
 * · **Wan 2.1 I2V-14B** — a real video transformer in `.venv-flux`. ~7 minutes,
 *   up to ~5s, and it renders at its own 480p pixel budget rather than yours.
 *   Any image qualifies (it reads pixels, not conditioning), including sdxl and
 *   FLUX upscales. It invents events — things enter the frame and move.
 */
import { Slider } from '../../components/ui/Slider'
import { Tool } from '../../components/ui/Tool'
import { useUI } from '../../store'

/** The motion module's trained window — past it the clip degenerates to noise. */
export const ANIM_MAX_FRAMES = 16
export const ANIM_DENOISE = 0.6
/** Wan's temporal VAE compresses time 4:1, so frames land on a 4k+1 grid. */
export const WAN_FRAMES = 49

export function AnimateBar() {
  const engine = useUI((s) => s.tools.anEngine)
  const source = useUI((s) => s.tools.anSource)
  const prompt = useUI((s) => s.tools.anPrompt)
  const wanPrompt = useUI((s) => s.tools.wnPrompt)
  const wanSeed = useUI((s) => s.tools.wnSeed)
  const setTool = useUI((s) => s.setTool)
  const wan = engine === 'wan'

  return (
    <>
      <Tool
        id="anEngine"
        label="Animate"
        tooltip={
          'AnimateDiff: ~20s, 2s of video, your resolution, sd15/sd2 only — drift and ' +
          'atmosphere. Wan: ~7min, up to 5s, 480p budget, any image — real motion.'
        }
        width="w-[150px]"
        options={[
          { value: 'animatediff', label: 'AnimateDiff' },
          { value: 'wan', label: 'Wan 2.1 I2V' },
        ]}
      />

      {wan ? (
        <>
          <Tool
            id="wnModel"
            label="Model"
            tooltip={
              '2.2 (A14B): mixture of experts — two 14B transformers, one for the ' +
              'noisy half of the schedule and one for the clean half. 2.1 (14B): ' +
              'a single older transformer at a 480p budget.'
            }
            width="w-[150px]"
            options={[
              { value: '2.2-a14b', label: 'Wan 2.2 (A14B)' },
              { value: '2.1-14b', label: 'Wan 2.1 (14B)' },
            ]}
          />
          <Tool
            id="wnLightning"
            label="Lightning"
            tooltip={
              'The lightx2v step-distillation LoRA: 6 steps instead of 30. Measured ' +
              'here at 60s vs 205s a clip, and the first frame came out CLOSER to ' +
              'the source, not further. Off only to see what it costs you.'
            }
            width="w-[110px]"
            options={[
              { value: '1', label: 'on (6 st)' },
              { value: '', label: 'off (30)' },
            ]}
          />
          <Slider
            id="wnFrames"
            label="Frames"
            tooltip="Snapped to 4k+1 by the temporal VAE. 49 ≈ 3s, 81 ≈ 5s."
            min={17}
            max={81}
            step={4}
            fallback={WAN_FRAMES}
            format={(v) => `${v} (~${(v / 16).toFixed(1)}s)`}
            width="w-[170px]"
          />
          <Tool
            id="wnSize"
            label="Size"
            tooltip={
              "Wan renders 480-832px. Auto follows its own rule (a square goes to " +
              '640). Matching your source’s size removes the resample — the ' +
              'biggest single lever on how much frame 1 looks like your picture.'
            }
            width="w-[130px]"
            options={[
              { value: '', label: 'auto (640)' },
              { value: '704', label: '704 px' },
              { value: '768', label: '768 px' },
              { value: '832', label: '832 px (max)' },
            ]}
          />
          <Tool
            id="wnQuant"
            label="Precision"
            tooltip={
              'nf4 crushes every weight to 4 bits — it is what makes two 14B ' +
              'experts fit at all. fp8 is the reference implementation\'s ' +
              'precision (8 bits) and visibly cleaner, paid for by swapping the ' +
              'inactive expert out to host RAM: slower per clip.'
            }
            width="w-[130px]"
            options={[
              { value: 'nf4', label: 'nf4 (fast)' },
              { value: 'fp8', label: 'fp8 (better)' },
            ]}
          />
          <Tool
            id="wnGuidance"
            label="Guidance"
            tooltip={
              'CFG. 1 = OFF, which is what Wan’s own reference uses and what keeps ' +
              'the first frame faithful. Higher extrapolates away from the negative ' +
              'toward the positive — with a motion-only prompt there is nothing to ' +
              'aim at, so it overpowers the image and redraws it.'
            }
            width="w-[130px]"
            options={[
              { value: '1', label: '1 — off' },
              { value: '2', label: '2' },
              { value: '3.5', label: '3.5' },
              { value: '5', label: '5 — drifts' },
            ]}
          />
        </>
      ) : (
        <>
          <Tool
            id="anSource"
            label="From"
            tooltip={
              'Pixels: img2img from the image you see — a flux upscale keeps its detail. ' +
              "Conditioning: re-render the image's own off-grid coordinate in motion."
            }
            width="w-[150px]"
            options={[
              { value: 'pixels', label: 'Pixels (img2img)' },
              { value: 'cond', label: 'Conditioning' },
            ]}
          />

          {source !== 'cond' ? (
            <Slider
              id="anDenoise"
              label="Denoise"
              tooltip={
                'How far the clip may depart from the source pixels. 0.6 balances; ' +
                'lower pins the picture and barely moves, higher moves more and drifts.'
              }
              min={0.2}
              max={0.95}
              step={0.05}
              fallback={ANIM_DENOISE}
              width="w-[150px]"
            />
          ) : null}

          <Slider
            id="anFrames"
            label="Frames"
            tooltip={
              `Hard ceiling ${ANIM_MAX_FRAMES}: the motion module was trained on that ` +
              'window and produces coloured noise past it. Lengthen with Smooth instead.'
            }
            min={8}
            max={ANIM_MAX_FRAMES}
            step={1}
            fallback={ANIM_MAX_FRAMES}
            format={(v) => `${v}`}
            width="w-[120px]"
          />

          {/* Conditioning mode only. FreeInit re-noises to the FULL noise level
              each pass, and Pixels mode truncates the schedule to denoise x steps
              — too few left to come back down, so the clip lands in confetti.
              Measured, not theorised: clean at denoise 0.6 without it, destroyed
              with it. */}
          {source === 'cond' ? (
            <Tool
              id="anFreeInit"
              label="Steady"
              tooltip={
                'FreeInit: re-run sampling, each pass reusing the previous clip\'s ' +
                'low-frequency structure — steadier motion at this many times the ' +
                'render cost. Needs the full schedule, so Conditioning only.'
              }
              width="w-[110px]"
              options={[
                { value: '0', label: 'off' },
                { value: '2', label: '×2' },
                { value: '3', label: '×3' },
                { value: '5', label: '×5' },
              ]}
            />
          ) : null}

          <Tool
            id="anLora"
            label="Camera"
            tooltip="A motion LoRA layered on the module: a deliberate camera move."
            width="w-[130px]"
            options={[
              { value: 'none', label: 'none' },
              { value: 'zoom-in', label: 'zoom in' },
              { value: 'zoom-out', label: 'zoom out' },
              { value: 'pan-left', label: 'pan left' },
              { value: 'pan-right', label: 'pan right' },
              { value: 'tilt-up', label: 'tilt up' },
              { value: 'tilt-down', label: 'tilt down' },
              { value: 'roll-cw', label: 'roll ↻' },
              { value: 'roll-ccw', label: 'roll ↺' },
            ]}
          />
        </>
      )}

      <input
        className="sa-input !w-[230px] !py-[5px] !text-[12px]"
        placeholder={wan ? 'motion prompt (blank = Wan default)' : 'motion prompt (optional)'}
        value={(wan ? wanPrompt : prompt) ?? ''}
        onChange={(e) => setTool(wan ? 'wnPrompt' : 'anPrompt', e.target.value)}
        title={
          wan
            ? 'Describe the MOTION, never the content — the image stays the only ' +
              'description of what the thing is. Blank uses Wan’s own default ' +
              '(“make this image come alive, cinematic motion, smooth animation”).'
            : "AnimateDiff's temporal layers were trained on text; off-grid " +
              'conditioning gives them little to enact. Describing the MOTION ' +
              '(not the picture) is what makes a clip move instead of shimmer.'
        }
      />

      <input
        className="sa-input !w-[110px] !py-[5px] !text-[12px]"
        placeholder="seed"
        value={(wan ? wanSeed : '') ?? ''}
        onChange={(e) => setTool('wnSeed', e.target.value)}
        disabled={!wan}
        title="blank = a fresh random seed per clip (always recorded in film.json). Pin one to reproduce a clip, or to A/B a knob against identical noise."
      />

      <Tool
        id="anSmooth"
        label="Smooth"
        tooltip={
          'Also write a motion-interpolated copy at this frame rate — it invents the ' +
          'in-betweens rather than asking the model for frames it cannot draw.'
        }
        width="w-[110px]"
        options={[
          { value: '0', label: 'off' },
          { value: '16', label: '16 fps' },
          { value: '24', label: '24 fps' },
          { value: '30', label: '30 fps' },
        ]}
      />
    </>
  )
}
