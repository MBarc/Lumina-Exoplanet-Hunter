# Exoplanet Hunter

**A distributed computing network for the detection of exoplanet candidates across open astronomical datasets.**

---

## What Is This?

Exoplanet Hunter is an open-source project that turns idle computers into exoplanet search nodes. Every participating machine downloads, processes, and analyzes stellar light curve data from NASA and other space missions — automatically, in the background, while you go about your day.

Together, these machines form **ExoNet**: a volunteer-powered network working toward a single goal — finding worlds beyond our solar system.

---

## The Problem

Space telescopes like TESS, Kepler, and K2 have produced an enormous archive of stellar light curve data. Hidden within that data are the faint, periodic dimming signatures of planets transiting their host stars. The archive grows faster than it can be analyzed.

There are more potential exoplanet candidates waiting in existing data than current resources allow us to find.

---

## How It Works

1. **Install** the Exoplanet Hunter client on any Windows machine
2. The client runs quietly as a **background service** — no interaction required
3. It connects to the ExoNet coordination network, claims an unprocessed data sector, and begins analysis
4. Light curves are retrieved from mission archives (TESS, Kepler, K2, and others), processed locally, and screened for transit signatures
5. Candidate detections are reported back to the network for further review

The more machines participating, the faster the full archive gets covered.

---

## Supported Missions

Exoplanet Hunter is designed to work with any mission that produces light curve data, including:

- **TESS** (Transiting Exoplanet Survey Satellite)
- **Kepler**
- **K2**
- Additional missions as support is added

---

## Getting Started

Download the installer and run it. That's it.

The installer will configure your machine, set up the background service, and connect you to the ExoNet network automatically. No astronomy background required.

> *Installer and setup instructions coming soon.*

---

## Local Dashboard

Every ExoNet node includes a locally hosted web dashboard accessible from your browser. No account or internet connection required to view it — it runs entirely on your machine.

The dashboard lets you see:

- Which mission and sector your machine is currently processing
- How many stars have been analyzed and how many remain
- A live feed of light curves as they are processed, with transit detections highlighted
- Any candidate signals your machine has flagged for review
- Your node's contribution to the broader ExoNet network over time

It is designed to be left open in a browser tab — something you can glance at while working.

---

## If Your Machine Finds a Candidate

When your node detects a statistically significant transit signal, you will be notified through the dashboard. You will also have the opportunity to assign a **nickname** to the candidate — a name that will be associated with it permanently within ExoNet.

If the candidate is later confirmed as a genuine exoplanet through follow-up observation, that nickname becomes your nomination for the planet's official name.

Official exoplanet naming is governed by the **International Astronomical Union (IAU)**, which periodically runs public naming campaigns (NameExoWorlds) for confirmed exoplanets. ExoNet does not guarantee official recognition, but confirmed candidates discovered through this project will be submitted through proper IAU channels with the discoverer's nominated name on record.

The universe is large. Your name could end up on a world orbiting another star.

---

## For Researchers & Developers

Exoplanet Hunter is fully open source. If you are interested in contributing to the detection pipeline, extending mission support, or integrating candidate data into your own research workflows, see the project source and open an issue or pull request.

---

## Why This Matters

Every candidate flagged by ExoNet is a star worth a closer look — a potential system with a planet in orbit, possibly within a habitable zone. The data is already out there. This project exists to make sure none of it goes unexamined.

---

*Exoplanet Hunter is an independent open-source initiative and is not affiliated with NASA or any space agency.*
