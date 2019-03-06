#ifndef AUX_HPP
#define AUX_HPP

#include <chrono>
#include <algorithm>

namespace aux {

template <class T>
void endswap(T& objp)
{
    unsigned char *memp = reinterpret_cast<unsigned char*>(&objp);
    std::reverse(memp, memp + sizeof(T));
}


class Timer
{
private:
	// Type aliases to make accessing nested type easier
	using clock_t = std::chrono::high_resolution_clock;
	using second_t = std::chrono::duration<double, std::ratio<1> >;
	
	std::chrono::time_point<clock_t> m_beg;
 
public:
	Timer() : m_beg(clock_t::now()) {}
	
	void reset() { m_beg = clock_t::now(); }
	
	double elapsed() const
	{
		return std::chrono::duration_cast<second_t>(clock_t::now()
                                                    - m_beg).count();
	}
    
    void report() const
    {
        std::cout << "Elapsed time: " << this->elapsed() << " seconds.\n";
    }
};

// namespace
}

#endif
