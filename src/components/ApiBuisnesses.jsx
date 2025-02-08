import { useState } from "react";
import Navbar from "./Navbar/navbar2";

export default function API_Buisnesses() {
    const [LoaderActive, setLoaderActive] = useState(false);

    return (
        <div className="relative flex flex-col justify-center items-center ">
            <Navbar active={3}/>
            {LoaderActive && (
                <div className="absolute inset-0 z-50 flex justify-center items-center bg-black bg-opacity-60">
                    <Loader size={64} className="animate-spin text-white" />
                    <div className="text-white text-2xl font-semibold ml-4">Loading ...</div>
                </div>
            )}
            {/* Aligning heading with navbar */}
            <div className={`w-[85%] pt-8 px-12  mt-20 ${LoaderActive ? "opacity-50" : ""}`}>
                <h1 className="text-3xl font-semibold">API for Businesses</h1>
                <p className="text-2xl pt-2 font-light">Empower your platform with Deepfake Detection and Blockchain Verification</p>
            </div>
            <div className={`w-[85%] px-12 mt-20 ${LoaderActive ? "opacity-50" : ""}`}>
                <h1 className="text-3xl font-semibold">Why choose DeepTrace API ?</h1>
                    <div className="flex gap-x-20 pt-6">
                        <div className="p-8 rounded-2xl bg-[#252525]">
                            <div>
                                <div className="text-2xl font-semibold">
                                State-of-the-Art AI <br /> Detection
                                </div>
                                <div className="pt-2 text-xl font-light">
                                    Detect AI-generated <br/> deepfakes with high accuracy <br/> using multi-modal analysis.
                                </div>
                            </div>
                        </div>
                        <div className="p-8 rounded-2xl bg-[#252525]">
                        <div>
                                <div className="text-2xl font-semibold">
                                Blockchain-powered <br /> Verification
                                </div>
                                <div className="pt-2 text-xl font-light">
                                    Securely store and verify <br/> content authenticity using <br/> Hyperledger technology.
                                </div>
                            </div>
                        </div>
                        <div className="p-8 rounded-2xl bg-[#252525]">
                        <div>
                                <div className="text-2xl font-semibold">
                                Customizable & <br /> Adaptive
                                </div>
                                <div className="pt-2 text-xl font-light">
                                    Custom APIs specially curated <br/> as per the custom buisness <br/> databases.
                                </div>
                            </div>
                        </div>
                    </div>
            </div>
            <div className={`w-[85%] px-12 mt-20 ${LoaderActive ? "opacity-50" : ""}`}>
                <h1 className="text-3xl font-semibold">How it works ?</h1>
                <div className="flex gap-x-20 pt-6">
                    <div className="p-8 rounded-2xl bg-[#252525]">
                        <div>
                            <div className="text-2xl font-semibold">
                            Upload Video/Image
                            </div>
                            <div className="pt-2 text-xl font-light">
                            Submit a video or image <br/> via API for <br/> deepfake analysis.
                            </div>
                        </div>
                    </div>
                    <div className="p-8 rounded-2xl bg-[#252525]">
                    <div>
                            <div className="text-2xl font-semibold">
                            AI & Metadata <br />  Scanning
                            </div>
                            <div className="pt-2 text-xl font-light">
                            Our AI models analyze <br/> pixel data, metadata, and <br/> tampering patterns.
                            </div>
                        </div>
                    </div>
                    <div className="p-8 rounded-2xl bg-[#252525]">
                    <div>
                            <div className="text-2xl font-semibold">
                            Blockchain proof of <br /> Authenticity
                            </div>
                            <div className="pt-2 text-xl font-light">
                            Check if the video already <br /> exists in our blockchain <br/> ledger or create a new <br/> authenticity record.
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <div className={`w-[85%] px-12 mt-20 ${LoaderActive ? "opacity-50" : ""}`}>
                <h1 className="text-3xl font-semibold">Use Cases</h1>
                <div className="flex gap-x-20 pt-6">
                    <div className="p-8 rounded-2xl bg-[#252525]">
                        <div>
                            <div className="text-2xl font-semibold">
                            Social media & <br/> Content platforms
                            </div>
                            <div className="pt-2 text-xl font-light">
                            Automatically flag <br/> deepfake content before <br/> it spreads.
                            </div>
                        </div>
                    </div>
                    <div className="p-8 rounded-2xl bg-[#252525]">
                    <div>
                            <div className="text-2xl font-semibold">
                            News & Media <br /> Organizations
                            </div>
                            <div className="pt-2 text-xl font-light">
                            Verify the authenticity  <br/> of breaking news footage  <br/> before publication.
                            </div>
                        </div>
                    </div>
                    <div className="p-8 rounded-2xl bg-[#252525]">
                    <div>
                            <div className="text-2xl font-semibold">
                            Government & Law <br /> Enforcement
                            </div>
                            <div className="pt-2 text-xl font-light">
                            Detect manipulated content <br /> in forensic investigations.
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <div className={`w-[85%] px-12 mt-20 ${LoaderActive ? "opacity-50" : ""}`}>
                <h1 className="text-3xl font-semibold">API features</h1>
                <div className="flex gap-x-20 pt-6">
                    <div className="p-8 rounded-2xl bg-[#252525]">
                        <div>
                            <div className="text-2xl font-semibold">
                            Deepfake probability score
                            </div>
                            <div className="pt-2 text-xl font-light">
                            Get a confidence <br/> score for video <br/> authenticity.
                            </div>
                        </div>
                    </div>
                    <div className="p-8 rounded-2xl bg-[#252525]">
                    <div>
                            <div className="text-2xl font-semibold">
                            Frame-by-Frame <br />  Analysis
                            </div>
                            <div className="pt-2 text-xl font-light">
                            Identify specific tampered <br/> frames in a video.
                            </div>
                        </div>
                    </div>
                    <div className="p-8 rounded-2xl bg-[#252525]">
                    <div>
                            <div className="text-2xl font-semibold">
                            Metadata Extraction
                            </div>
                            <div className="pt-2 text-xl font-light">
                            Analyze EXIF data,  <br /> C2PA, and <br/> watermark traces.
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <div className={`w-[85%] px-12 mt-20 ${LoaderActive ? "opacity-50" : ""}`}>
                <h1 className="text-3xl font-semibold">Pricing Plans</h1>
                <div className="flex gap-x-16 pt-6">
                    <div className="p-6 rounded-2xl bg-[#252525]">
                        <div>
                            <div className="text-2xl font-semibold">
                            Enterprise
                            </div>
                            <div className="pt-2 text-xl font-light">
                            Large scale API  <br/>usage with dedicated <br/> support & SLAs.
                            </div>
                        </div>
                    </div>
                    <div className="p-6 rounded-2xl bg-[#252525]">
                        <div>
                            <div className="text-2xl font-semibold">
                            Pro
                            </div>
                            <div className="pt-2 text-xl font-light">
                            Medium scale APIs<br/> access for buisnesses <br/> & agencies.
                            </div>
                        </div>
                    </div>
                    <div className="p-6 rounded-2xl bg-[#252525]">
                        <div>
                            <div className="text-2xl font-semibold">
                            Startup
                            </div>
                            <div className="pt-2 text-xl font-light">
                            Affordable API access <br /> for eaarly stage <br/> projects.]
                            </div>
                        </div>
                    </div>
                    <div className="p-6 rounded-2xl bg-[#252525]">
                        <div>
                            <div className="text-2xl font-semibold">
                                Free Trial
                            </div>
                            <div className="pt-2 text-xl font-light">
                            Test our API with <br/> limited requests.
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
