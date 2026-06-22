**readme**

Problem (see concept)

Solution: architecture

How
- Data story
- Eval: methodology and results
- see process for full

How to run
- git clone, add .env file, docker compose up, open localhost, score

Limitations
- only evaluated on a small set and for one area in sg
- synthetic data: only generated 7 images, quality can be further tuned.
- data/label quality: single human rater, poor image quality (even human struggles to tell the shade), sidewalks are sometimes obstructed or to the side
- calibration: in some instances the reasoning implies that the score ought to be lower, high confidence in all images
- free tier: limited to 15 per batch, inference is slow
- scalability issues: manual evaluation, API reliance
- did not evaluate shade source

Deployment section
- who would run this: government agency (nparks, URA) with a set of images and metadata. alternatively, a dropped idea was that citizens could use it to map out a shaded route
- monitoring: track API failure rate, consider a classifier to label important categories for analysis, consider collecting feedback from users about whether they agree with the score, prediction latency, flag low confidence scores for human review
- risk: predictions may be based on the wrong time of day, outdated, or not based on a walkable path

Demo video

**process**
- dataset curation: combination of reading website + initial pass through AI, then tune by exploring data columns to understand what to filter.
- handcrafted architecture after brainstorming with AI and some own exploration (see concept.md)
- reviewed code and tests while they are being written, had AI do a code review, continued thinking about test gaps/guardrails, code quality (e.g. refactoring to ensure eval uses existing code)
- noticed inference was slow, add parallel
- initial evaluation approach involved sky view index and green view index as baseline comparison, but not very sound. pivoted to consistency study instead
- initially wanted to use place metadata to conduct error analysis, but inspection showed that it is inaccurate (e.g. labelling grass as picnic area), pivoted to manual labelling of category
- noticed that certain categories (e.g. walkways, buildings) were underrepresented -> image generation to cover edge cases
- wanted to write a script to run image generation, but no free API (at least not within gemini. since this was not the focus, didn't delve too much into it). therefore manually generated with nano banana
- initial idea was to create a way to map out shaded routes. out of scope for 2 day project
- handcrafted result interpretation, limitations

