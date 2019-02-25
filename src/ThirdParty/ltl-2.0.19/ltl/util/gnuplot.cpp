/*
Copyright (c) 2009 Daniel Stahlke

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <ltl/util/gnuplot.h>

#include <stdexcept>
#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>

#if defined(HAVE_PTY_H)
#  include <pty.h>
#else
#  if defined(HAVE_UTIL_H)
#    include <util.h>
#  endif
#endif

namespace ltl {

Gnuplot::Gnuplot(const std::string cmd) :
   //ltl::fdostream(fileno(pout = popen(cmd.c_str(), "w"))),
   ltl::fdostream(),
   pty_fh(NULL),
   master_fd(-1),
   slave_fd(-1),
   debug_messages(false)
{
   string cmd_ = cmd;
   if( cmd_ == "" )
   {
      cmd_ = "gnuplot";
      char* envcmd = getenv("LTL_GNUPLOT");
      if( envcmd != 0 )
         cmd_ = envcmd;
   }

   init(fileno(pout = popen(cmd_.c_str(), "w")));

	setf(std::ios::scientific, std::ios::floatfield);
	precision(18);
}

Gnuplot::~Gnuplot() {
	if(debug_messages) {
		std::cerr << "closing gnuplot" << std::endl;
	}

	pclose(pout);
	if(pty_fh) fclose(pty_fh);
	if(master_fd > 0) ::close(master_fd);
	if(slave_fd  > 0) ::close(slave_fd);
}

void Gnuplot::getMouse(double &mx, double &my, int &mb) {
	allocReader();
	*this << "pause mouse \"Click mouse!\\n\"" << std::endl;
	*this << "print MOUSE_X, MOUSE_Y, MOUSE_BUTTON" << std::endl;
	if(debug_messages) {
		std::cerr << "begin scanf" << std::endl;
	}
	if(3 != fscanf(pty_fh, "%lf %lf %d", &mx, &my, &mb)) {
		throw std::runtime_error("could not parse reply");
	}
	if(debug_messages) {
		std::cerr << "end scanf" << std::endl;
	}
}

// adapted from http://www.gnuplot.info/files/gpReadMouseTest.c
void Gnuplot::allocReader() {
	if(pty_fh) return;

	if(0 > openpty(&master_fd, &slave_fd, NULL, NULL, NULL)) {
		perror("openpty");
		throw std::runtime_error("openpty failed");
	}
	char pty_fn_buf[1024];
	if(ttyname_r(slave_fd, pty_fn_buf, 1024)) {
		perror("ttyname_r");
		throw std::runtime_error("ttyname failed");
	}
	pty_fn = std::string(pty_fn_buf);
	if(debug_messages) {
		std::cerr << "fn=" << pty_fn << std::endl;
	}

	// disable echo
	struct termios tios;
	if(tcgetattr(slave_fd, &tios) < 0) {
		perror("tcgetattr");
		throw std::runtime_error("tcgetattr failed");
	}
	tios.c_lflag &= ~(ECHO | ECHONL);
	if(tcsetattr(slave_fd, TCSAFLUSH, &tios) < 0) {
		perror("tcsetattr");
		throw std::runtime_error("tcsetattr failed");
	}

	pty_fh = fdopen(master_fd, "r");
	if(!pty_fh) {
		throw std::runtime_error("fdopen failed");
	}

	*this << "set mouse; set print \"" << pty_fn << "\"" << std::endl;
}

void Gnuplot::interactive( const string& pname )
{
   cout<<pname<<" gnuplot > ";
   string command, param;
   command = "";
   param = "";
   std::cin>>command;
   while (command!="quit"&&command.substr(0,1)!="q")
   {
      if (command.substr (0, 1)=="p")
      {
         getline(std::cin,param);
         if (param=="")
         {
            cout<<"Usage: print filename.pdf";
            cout<<"pname gnuplot > ";
            std::cin>>command;
            continue;
         }
         const string outname = param.substr( param.find_first_not_of( " "));
         cout << "Writing plot to "<<outname<<endl;
         *this<<"set term pdf"<<endl;
         *this<<"set output '"<<outname<<"'"<<endl;
         *this<<"refresh"<<endl;
      }
      else if (command.substr (0, 7)=="refresh")
      {
         getline(std::cin,param);
         *this << "refresh" << endl;
      }
      else if (command.substr (0, 2)=="gp")
      {
         getline(std::cin,param);
         cout << "Sending to gnuplot"<<endl;
         *this << param << endl;
      }
      else
      {
      cout<<"Unknown command, use quit or print!"<<endl;
      }
      cout<<pname<<" gnuplot > ";
      std::cin>>command;
   }

   return;
}

}
